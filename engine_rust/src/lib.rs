//! Market Twin • Motor Nativo de Simulação em Rust (Rayon + C-ABI)
//! Geração estocástica e cálculo micro-econômico paralelo ultra-rápido para até 1.000.000 de personas.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Beta, LogNormal, Normal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct NativePitchInput {
    pub name: String,
    pub category: String,
    pub price_usd_monthly: f64,
    pub target_regions: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegionalMetricOut {
    pub region: String,
    pub total_population: u64,
    pub converted_users: u64,
    pub conversion_rate: f64,
    pub average_pmf: f64,
    pub average_wtp_usd: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PricePointOut {
    pub price_usd: f64,
    pub conversion_rate: f64,
    pub projected_revenue_index: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeSimOutput {
    pub population_size: u64,
    pub pmf_index: f64,
    pub nps_score: f64,
    pub conversion_rate: f64,
    pub duration_ms: f64,
    pub breakdown_by_region: Vec<RegionalMetricOut>,
    pub price_elasticity: Vec<PricePointOut>,
}

#[inline(always)]
fn get_category_affinity(category: &str, archetype_idx: usize) -> f64 {
    // Archetype Indices:
    // 0: Enterprise Buyer
    // 1: SMB Founder
    // 2: Freelancer Dev
    // 3: Tech Enthusiast
    // 4: Everyday Consumer
    // 5: Student
    match category {
        "DevTool / Infra" => match archetype_idx {
            2 => 0.95,
            3 => 0.85,
            0 => 0.65,
            1 => 0.60,
            5 => 0.55,
            _ => 0.10,
        },
        "AI Agent / Automation" => match archetype_idx {
            1 => 0.92,
            3 => 0.90,
            2 => 0.85,
            0 => 0.75,
            5 => 0.65,
            _ => 0.60,
        },
        "FinTech / Payments" => match archetype_idx {
            1 => 0.90,
            0 => 0.85,
            4 => 0.75,
            2 => 0.70,
            3 => 0.65,
            _ => 0.50,
        },
        "Consumer App / Mobile" => match archetype_idx {
            4 => 0.90,
            5 => 0.85,
            3 => 0.80,
            2 => 0.60,
            1 => 0.40,
            _ => 0.25,
        },
        "EdTech / Learning" => match archetype_idx {
            5 => 0.95,
            2 => 0.80,
            4 => 0.70,
            3 => 0.65,
            1 => 0.50,
            _ => 0.45,
        },
        // Padrão: SaaS B2B
        _ => match archetype_idx {
            0 => 0.90,
            1 => 0.85,
            2 => 0.65,
            3 => 0.50,
            5 => 0.20,
            _ => 0.15,
        },
    }
}

pub fn simulate_fast(
    input: &NativePitchInput,
    population_size: u64,
    seed: u64,
) -> NativeSimOutput {
    let start = std::time::Instant::now();

    let region_names = [
        "LATAM",
        "North America",
        "Europe",
        "Asia-Pacific",
        "Middle East & Africa",
    ];
    let region_weights = [0.35, 0.25, 0.20, 0.15, 0.05];
    let income_mus = [6.8, 8.4, 8.0, 7.4, 6.7];
    let income_sigmas = [0.65, 0.55, 0.50, 0.75, 0.85];

    let archetype_weights = [0.10, 0.20, 0.25, 0.15, 0.20, 0.10];

    let base_price = input.price_usd_monthly.max(1.0);

    // Multiplicadores da Curva de Elasticidade
    let elasticity_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0];
    let mut elasticity_prices = [0.0; 8];
    for (i, &m) in elasticity_multipliers.iter().enumerate() {
        elasticity_prices[i] = (base_price * m * 100.0).round() / 100.0;
    }

    // Estrutura de Acumulação Paralela por Chunk de Thread
    struct LocalStats {
        buyers: u64,
        promoters: u64,
        detractors: u64,
        pmf_sum: f64,
        reg_totals: [u64; 5],
        reg_buyers: [u64; 5],
        reg_pmf_sum: [f64; 5],
        reg_wtp_sum: [f64; 5],
        elasticity_converted: [u64; 8],
    }

    impl LocalStats {
        fn new() -> Self {
            Self {
                buyers: 0,
                promoters: 0,
                detractors: 0,
                pmf_sum: 0.0,
                reg_totals: [0; 5],
                reg_buyers: [0; 5],
                reg_pmf_sum: [0.0; 5],
                reg_wtp_sum: [0.0; 5],
                elasticity_converted: [0; 8],
            }
        }

        fn merge(&mut self, other: &Self) {
            self.buyers += other.buyers;
            self.promoters += other.promoters;
            self.detractors += other.detractors;
            self.pmf_sum += other.pmf_sum;
            for i in 0..5 {
                self.reg_totals[i] += other.reg_totals[i];
                self.reg_buyers[i] += other.reg_buyers[i];
                self.reg_pmf_sum[i] += other.reg_pmf_sum[i];
                self.reg_wtp_sum[i] += other.reg_wtp_sum[i];
            }
            for i in 0..8 {
                self.elasticity_converted[i] += other.elasticity_converted[i];
            }
        }
    }

    // Processamento Paralelo via Rayon
    let chunk_size = 4096;
    let num_chunks = ((population_size as f64) / (chunk_size as f64)).ceil() as u64;

    let combined_stats = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(population_size);
            let mut local = LocalStats::new();

            // Gerador determinístico baseado na semente e chunk
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(chunk_idx * 10007));
            let beta_sav = Beta::new(2.5, 2.5).unwrap();
            let beta_skep = Beta::new(3.0, 2.5).unwrap();
            let normal_noise = Normal::new(0.0, 0.15).unwrap();
            let normal_sens = Normal::new(0.0, 0.08).unwrap();

            for _i in chunk_start..chunk_end {
                // Sorteio de Região
                let r_val: f64 = rng.gen();
                let mut reg_idx = 0;
                let mut cum = 0.0;
                for (idx, &w) in region_weights.iter().enumerate() {
                    cum += w;
                    if r_val <= cum {
                        reg_idx = idx;
                        break;
                    }
                }

                // Sorteio de Arquétipo
                let a_val: f64 = rng.gen();
                let mut arch_idx = 0;
                let mut a_cum = 0.0;
                for (idx, &w) in archetype_weights.iter().enumerate() {
                    a_cum += w;
                    if a_val <= a_cum {
                        arch_idx = idx;
                        break;
                    }
                }

                // Tech Savviness & Ceticismo
                let mut savviness: f64 = rng.sample(beta_sav);
                if arch_idx == 2 || arch_idx == 3 {
                    savviness = (savviness * 1.35 + 0.15).min(1.0);
                } else if arch_idx == 0 {
                    savviness = (savviness * 1.1 + 0.05).min(1.0);
                }

                let mut skepticism: f64 = rng.sample(beta_skep);
                if arch_idx == 0 {
                    skepticism = (skepticism * 1.25 + 0.1).min(1.0);
                }

                // Renda Log-Normal
                let log_dist = LogNormal::new(income_mus[reg_idx], income_sigmas[reg_idx]).unwrap();
                let mut income: f64 = rng.sample(log_dist);

                if arch_idx == 0 {
                    income *= rng.gen_range(3.5..7.0);
                } else if arch_idx == 1 {
                    income *= rng.gen_range(2.0..4.0);
                } else if arch_idx == 5 {
                    income *= rng.gen_range(0.3..0.6);
                }

                // Sensibilidade
                let norm_income = (income / 10000.0).clamp(0.05, 1.0);
                let base_sens = 1.0 - (norm_income.ln_1p() / 2.0_f64.ln_1p()) * 0.7;
                let sens: f64 = (base_sens + rng.sample(normal_sens)).clamp(0.05, 0.98);

                // Afinidade e WTP
                let affinity = get_category_affinity(&input.category, arch_idx);
                let base_wtp = (income * 0.05) * affinity;
                let income_factor = (income / 500.0).ln_1p();
                let max_wtp = (base_wtp * income_factor * (1.2 - sens * 0.6)).clamp(2.0, 15000.0);

                // Utilidade Micro-Econômica
                let pvr = max_wtp / base_price;
                let noise: f64 = rng.sample(normal_noise);
                let utility = affinity * 0.40
                    + (pvr / 2.0).min(1.0) * 0.35
                    + savviness * 0.15
                    - skepticism * 0.25
                    + noise;

                let would_buy = utility >= 0.48 && pvr >= 0.80;

                let raw_pmf = (utility * 10.0) + (pvr * 1.5);
                let pmf_score = raw_pmf.clamp(0.0, 10.0);

                // Atualizar agregados
                local.reg_totals[reg_idx] += 1;
                local.pmf_sum += pmf_score;
                local.reg_pmf_sum[reg_idx] += pmf_score;
                local.reg_wtp_sum[reg_idx] += max_wtp;

                if would_buy {
                    local.buyers += 1;
                    local.reg_buyers[reg_idx] += 1;
                }

                if pmf_score >= 8.5 && would_buy {
                    local.promoters += 1;
                } else if pmf_score < 6.5 {
                    local.detractors += 1;
                }

                // Elasticidade nos 8 preços
                for (p_idx, &test_p) in elasticity_prices.iter().enumerate() {
                    if test_p <= max_wtp && (max_wtp / test_p) >= 0.85 {
                        local.elasticity_converted[p_idx] += 1;
                    }
                }
            }

            local
        })
        .reduce(LocalStats::new, |mut a, b| {
            a.merge(&b);
            a
        });

    let n = population_size as f64;
    let conversion_rate = (combined_stats.buyers as f64 / n) * 100.0;
    let nps = ((combined_stats.promoters as f64 - combined_stats.detractors as f64) / n) * 100.0;
    let mean_pmf = combined_stats.pmf_sum / n;
    let pmf_index = ((mean_pmf * 7.0) + (conversion_rate * 0.30)).min(100.0);

    // Métricas por Região
    let mut regional_out = Vec::with_capacity(5);
    for i in 0..5 {
        let tot = combined_stats.reg_totals[i];
        let tot_f = tot.max(1) as f64;
        let c_rate = (combined_stats.reg_buyers[i] as f64 / tot_f) * 100.0;
        let avg_pmf = combined_stats.reg_pmf_sum[i] / tot_f;
        let avg_wtp = combined_stats.reg_wtp_sum[i] / tot_f;

        regional_out.push(RegionalMetricOut {
            region: region_names[i].to_string(),
            total_population: tot,
            converted_users: combined_stats.reg_buyers[i],
            conversion_rate: (c_rate * 100.0).round() / 100.0,
            average_pmf: (avg_pmf * 100.0).round() / 100.0,
            average_wtp_usd: (avg_wtp * 100.0).round() / 100.0,
        });
    }
    regional_out.sort_by(|a, b| b.conversion_rate.partial_cmp(&a.conversion_rate).unwrap());

    // Curva de Elasticidade
    let mut max_rev = 1.0_f64;
    let mut raw_elasticity = Vec::with_capacity(8);
    for (i, &p_usd) in elasticity_prices.iter().enumerate() {
        let conv_count = combined_stats.elasticity_converted[i];
        let c_rate = (conv_count as f64 / n) * 100.0;
        let rev = (conv_count as f64) * p_usd;
        if rev > max_rev {
            max_rev = rev;
        }
        raw_elasticity.push((p_usd, c_rate, rev));
    }

    let mut price_elasticity = Vec::with_capacity(8);
    for (p_usd, c_rate, rev) in raw_elasticity {
        let rev_idx = ((rev / max_rev) * 100.0 * 10.0).round() / 10.0;
        price_elasticity.push(PricePointOut {
            price_usd: p_usd,
            conversion_rate: (c_rate * 100.0).round() / 100.0,
            projected_revenue_index: rev_idx,
        });
    }

    let elapsed = start.elapsed();
    let duration_ms = elapsed.as_secs_f64() * 1000.0;

    NativeSimOutput {
        population_size,
        pmf_index: (pmf_index * 10.0).round() / 10.0,
        nps_score: (nps * 10.0).round() / 10.0,
        conversion_rate: (conversion_rate * 100.0).round() / 100.0,
        duration_ms: (duration_ms * 100.0).round() / 100.0,
        breakdown_by_region: regional_out,
        price_elasticity,
    }
}

// =========================================================================
// C-ABI EXPORTS (Zero-Overhead In-Memory Calls from Python ctypes)
// =========================================================================

#[no_mangle]
pub extern "C" fn run_native_simulation(
    input_json: *const c_char,
    population_size: u64,
    seed: u64,
) -> *mut c_char {
    if input_json.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CStr::from_ptr(input_json) };
    let json_slice = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let pitch_input: NativePitchInput = match serde_json::from_str(json_slice) {
        Ok(input) => input,
        Err(_) => return std::ptr::null_mut(),
    };

    let result = simulate_fast(&pitch_input, population_size, seed);

    let output_json = match serde_json::to_string(&result) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match CString::new(output_json) {
        Ok(c_res) => c_res.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn free_native_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_simulation_100k_performance() {
        let input = NativePitchInput {
            name: "PIX API".to_string(),
            category: "FinTech / Payments".to_string(),
            price_usd_monthly: 49.0,
            target_regions: None,
        };

        let result = simulate_fast(&input, 100_000, 42);
        println!(
            "100.000 personas simuladas em: {:.2}ms (Taxa Conversão: {:.2}%, PMF: {:.1})",
            result.duration_ms, result.conversion_rate, result.pmf_index
        );

        assert_eq!(result.population_size, 100_000);
        assert!(result.conversion_rate > 5.0);
        assert!(result.pmf_index > 0.0);
        assert!(result.duration_ms < 200.0, "Deveria rodar em menos de 200ms!");
    }
}
