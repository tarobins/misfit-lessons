r"""
wald_sprt_example.py
====================
A complete, self-contained demonstration of Abraham Wald's Sequential
Probability Ratio Test (SPRT, 1945) and its direct connection to
Information Theory and Relative Entropy (Kullback-Leibler Divergence).

Theoretical Background:
-----------------------
In classical fixed-sample hypothesis testing (Neyman-Pearson), sample size N is
fixed in advance. In contrast, Abraham Wald's SPRT collects observations
sequentially and evaluates the cumulative Log-Likelihood Ratio (LLR):

    S_n = \sum_{i=1}^n \ln \left( \frac{p(x_i \mid H_1)}{p(x_i \mid H_0)} \right) = S_{n-1} + z_i

Decision Boundaries:
    - If S_n >= \ln(B) \approx \ln((1 - \beta) / \alpha): Terminate, ACCEPT H_1
    - If S_n <= \ln(A) \approx \ln(\beta / (1 - \alpha)): Terminate, ACCEPT H_0
    - Otherwise (ln(A) < S_n < ln(B)): Continue sampling (take observation n + 1)

Information-Theoretic Link (Wald's Identity & KL Divergence):
------------------------------------------------------------
Under H_1, the expected drift per sample is:
    E_{H_1}[z_i] = E_{H_1}\left[\ln \frac{p(X \mid H_1)}{p(X \mid H_0)}\right] = D_KL(P_1 || P_0) > 0

Under H_0, the expected drift per sample is:
    E_{H_0}[z_i] = -E_{H_0}\left[\ln \frac{p(X \mid H_0)}{p(X \mid H_1)}\right] = -D_KL(P_0 || P_1) < 0

By Wald's Equation, the Expected Sample Size (Average Sample Number, ASN) is:
    E_{H_1}[N] \approx \frac{(1 - \beta) \ln B + \beta \ln A}{D_KL(P_1 || P_0)}
    E_{H_0}[N] \approx \frac{\alpha \ln B + (1 - \alpha) \ln A}{-D_KL(P_0 || P_1)}

Compared to fixed-sample tests, SPRT typically requires ~50% fewer samples on average!
"""

import numpy as np
import matplotlib.pyplot as plt


def run_single_sprt_bernoulli(p0: float, p1: float, alpha: float, beta: float, true_p: float, max_steps: int = 1000):
    """
    Simulate a single run of Wald's SPRT for Bernoulli trials.
    
    H0: p = p0
    H1: p = p1 (p1 > p0)
    alpha: Type I error bound (P(accept H1 | H0))
    beta:  Type II error bound (P(accept H0 | H1))
    true_p: Ground-truth probability of generating a 1.
    """
    # Wald's boundary approximations
    b_upper = np.log((1.0 - beta) / alpha)
    a_lower = np.log(beta / (1.0 - alpha))
    
    # Precompute log-likelihood increments:
    # If x = 1: ln(p1 / p0)
    # If x = 0: ln((1 - p1) / (1 - p0))
    llr_1 = np.log(p1 / p0)
    llr_0 = np.log((1.0 - p1) / (1.0 - p0))
    
    llr_path = [0.0]
    decision = None
    
    for step in range(1, max_steps + 1):
        x = 1 if np.random.rand() < true_p else 0
        increment = llr_1 if x == 1 else llr_0
        current_llr = llr_path[-1] + increment
        llr_path.append(current_llr)
        
        if current_llr >= b_upper:
            decision = "H1"
            break
        elif current_llr <= a_lower:
            decision = "H0"
            break
            
    if decision is None:
        decision = "UNDECIDED"
        
    return decision, np.array(llr_path), a_lower, b_upper


def kl_bernoulli(p: float, q: float) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q) in nats for binary distributions."""
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def main():
    np.random.seed(42)
    
    # -------------------------------------------------------------
    # Scenario: A/B Testing / Quality Control / Anomaly Detection
    # Null Hypothesis H0: Defect rate p0 = 10% (0.10)
    # Alt  Hypothesis H1: Defect rate p1 = 25% (0.25)
    # Error thresholds: alpha = 0.05 (5%), beta = 0.05 (5%)
    # -------------------------------------------------------------
    p0 = 0.10
    p1 = 0.25
    alpha = 0.05
    beta = 0.05
    
    dkl_1_0 = kl_bernoulli(p1, p0) # Drift under H1
    dkl_0_1 = kl_bernoulli(p0, p1) # Drift under H0
    
    b_upper = np.log((1.0 - beta) / alpha)
    a_lower = np.log(beta / (1.0 - alpha))
    
    # Theoretical Average Sample Numbers (ASN) via Wald's Identity:
    asn_h1_theory = ((1.0 - beta) * b_upper + beta * a_lower) / dkl_1_0
    asn_h0_theory = (alpha * b_upper + (1.0 - alpha) * a_lower) / (-dkl_0_1)
    
    # Fixed-Sample Test equivalent sample size (approximate normal approximation)
    # n_fixed = ((z_alpha * sqrt(p0*(1-p0)) + z_beta * sqrt(p1*(1-p1))) / (p1 - p0))^2
    z_alpha = 1.645 # standard normal 95th percentile
    z_beta  = 1.645
    n_fixed = ((z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    
    print("=" * 70)
    print("ABRAHAM WALD'S SEQUENTIAL PROBABILITY RATIO TEST (SPRT)")
    print("=" * 70)
    print(f"Hypotheses:  H0: p = {p0:.2f}  vs.  H1: p = {p1:.2f}")
    print(f"Risk Bounds: Type I error (alpha) = {alpha*100:.1f}%, Type II error (beta) = {beta*100:.1f}%")
    print("-" * 70)
    print(f"Upper Decision Threshold (Accept H1): ln((1-b)/a) = +{b_upper:.4f}")
    print(f"Lower Decision Threshold (Accept H0): ln(b/(1-a)) = {a_lower:.4f}")
    print("-" * 70)
    print(f"KL Divergence D_KL(P1 || P0): {dkl_1_0:.4f} nats (Expected drift under H1)")
    print(f"KL Divergence D_KL(P0 || P1): {dkl_0_1:.4f} nats (Expected drift under H0)")
    print(f"Theoretical Average Sample Number under H1: {asn_h1_theory:.1f} samples")
    print(f"Theoretical Average Sample Number under H0: {asn_h0_theory:.1f} samples")
    print(f"Classical Fixed-Sample Size (Neyman-Pearson): ~{int(np.ceil(n_fixed))} samples")
    print(f"Sample efficiency gain of SPRT: ~{(1.0 - max(asn_h0_theory, asn_h1_theory)/n_fixed)*100:.1f}% sample reduction!")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # Monte Carlo Evaluation
    # -------------------------------------------------------------
    n_mc = 500
    lengths_h0, decisions_h0 = [], []
    lengths_h1, decisions_h1 = [], []
    
    for _ in range(n_mc):
        d0, p0_path, _, _ = run_single_sprt_bernoulli(p0, p1, alpha, beta, true_p=p0)
        decisions_h0.append(d0)
        lengths_h0.append(len(p0_path) - 1)
        
        d1, p1_path, _, _ = run_single_sprt_bernoulli(p0, p1, alpha, beta, true_p=p1)
        decisions_h1.append(d1)
        lengths_h1.append(len(p1_path) - 1)
        
    print(f"\nEmpirical Monte Carlo Results ({n_mc} simulations per hypothesis):")
    print(f"  When H0 is TRUE (p = {p0}):")
    print(f"    - Correctly decided H0: {np.mean([d == 'H0' for d in decisions_h0])*100:.2f}% (Empirical Type I Error = {np.mean([d == 'H1' for d in decisions_h0])*100:.2f}%)")
    print(f"    - Mean Sample Number:    {np.mean(lengths_h0):.1f} ± {np.std(lengths_h0):.1f} samples (Theory: {asn_h0_theory:.1f})")
    
    print(f"  When H1 is TRUE (p = {p1}):")
    print(f"    - Correctly decided H1: {np.mean([d == 'H1' for d in decisions_h1])*100:.2f}% (Empirical Type II Error = {np.mean([d == 'H0' for d in decisions_h1])*100:.2f}%)")
    print(f"    - Mean Sample Number:    {np.mean(lengths_h1):.1f} ± {np.std(lengths_h1):.1f} samples (Theory: {asn_h1_theory:.1f})")
    
    # -------------------------------------------------------------
    # Visualization: Plot individual sample trajectories and boundaries
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Trajectories when H1 is true
    for _ in range(12):
        _, path, _, _ = run_single_sprt_bernoulli(p0, p1, alpha, beta, true_p=p1)
        ax1.plot(path, alpha=0.6, lw=1.5)
    ax1.axhline(b_upper, color="forestgreen", linestyle="--", lw=2, label=f"Accept H1 (ln B = {b_upper:.2f})")
    ax1.axhline(a_lower, color="crimson", linestyle="--", lw=2, label=f"Accept H0 (ln A = {a_lower:.2f})")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax1.set_title("SPRT Trajectories when H1 is TRUE (Drift > 0)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Sample Step (n)")
    ax1.set_ylabel("Cumulative Log-Likelihood Ratio S_n")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    
    # Right: Trajectories when H0 is true
    for _ in range(12):
        _, path, _, _ = run_single_sprt_bernoulli(p0, p1, alpha, beta, true_p=p0)
        ax2.plot(path, alpha=0.6, lw=1.5)
    ax2.axhline(b_upper, color="forestgreen", linestyle="--", lw=2, label=f"Accept H1 (ln B = {b_upper:.2f})")
    ax2.axhline(a_lower, color="crimson", linestyle="--", lw=2, label=f"Accept H0 (ln A = {a_lower:.2f})")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_title("SPRT Trajectories when H0 is TRUE (Drift < 0)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sample Step (n)")
    ax2.set_ylabel("Cumulative Log-Likelihood Ratio S_n")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plot_path = "wald_sprt_trajectories.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nVisualization saved to {plot_path}")


if __name__ == "__main__":
    main()
