# 🌍 Global Supply Chain Optimisation — Groupe Élégance (Luxury Perfume)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PuLP](https://img.shields.io/badge/PuLP-MILP-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Dijkstra-green)
![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-500%2B%20Iterations-purple)
![UCC](https://img.shields.io/badge/UCC-MSc%20Business%20Analytics-darkgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Association:** University College Cork (UCC) — MSc Business Analytics | Module: IS6055 Prescriptive Analytics  
> **Skills:** Python · PuLP · NetworkX · MILP · Two-Stage Stochastic Programming · Monte Carlo Simulation · Dijkstra's Algorithm · Matplotlib · Seaborn

---

## Why This Matters
Most supply chain projects optimise cost alone. This model simultaneously
optimises profit, CO₂ emissions and demand uncertainty — and proves that 
a €17.4M/month profit is achievable while fully respecting net-zero targets. 
Every recommendation is backed by 500+ Monte Carlo iterations.

---

## 📌 Project Overview

**Groupe Élégance** is a luxury perfume company headquartered in Grasse, France, with global operations across procurement, production, warehousing, and international distribution. This project builds a **full prescriptive analytics pipeline** to optimise the company's supply chain under cost, sustainability, and demand uncertainty constraints.

**Core Business Questions:**
- How should Groupe Élégance allocate procurement, production, and logistics to maximise profit while respecting a 500,000 kg CO₂/month cap?
- Which delivery routes minimise cost vs. time for urgent vs. non-urgent shipments?
- How robust is the optimal plan under real-world demand, supply, and workforce uncertainty?

**Key Results (Optimal CO₂-Capped Scenario):**
- Net Profit: **€17.4M/month**
- Total Production: **87,410 units** (97.1% Bordeaux capacity utilisation)
- CO₂ Utilisation: **100% of 500,000 kg cap** (binding constraint)
- Monte Carlo VaR (5%): profit exceeds **€7.2M** in 95% of full-uncertainty scenarios

---

## 🗂️ Repository Structure & How Files Connect

```
📁 Global-Supply-Chain-Optimisation/
│
├── 📜 Supply_Chain_Optimisation_Groupe_Elegance.py          ← MAIN FILE: Full optimisation pipeline
│   │   Sections 1–12 covering all assignment parts
│   │   Reads: no external dataset needed (all data embedded from Tables 1–9)
│   │   Outputs: 14 chart PNGs saved to working directory
│
├── 📁 Visualisation_Charts/
│   ├── 📊 fig1_profit_trajectory.png
│   ├── 📊 fig2_co2_trajectory.png
│   ├── 📊 fig3_sensitivity.png
│   ├── 📊 fig4_monte_carlo.png
│   ├── 📊 fig5_supplier_pie.png
│   ├── 📊 fig6_factory_emissions.png
│   ├── 📊 fig7_cost_heatmap.png
│   ├── 📊 fig8_warehouse_load.png
│   ├── 📊 fig9_network_topology.png
│   ├── 📊 fig10_pareto_frontier.png
│   ├── 📊 fig11_fulfillment_heatmap.png
│   ├── 📊 fig12_co2_breakdown.png
│   ├── 📊 fig13_stochastic_SP.png
│   └── 📊 fig14_stochastic_penalties.png
│
├── 📄 Business_Problem_Statement.pdf      ← Assignment brief (Tables 1–9, rubric)
├── 📦 requirements.txt                    ← Python dependencies
└── 📜 LICENSE                             ← MIT License
```

### 🔗 Pipeline Flow

```
Tables 1–9 (embedded data)
│
▼
[Section 2]  Network Graph + Dijkstra
             Cost-optimised routing (urgency ≤ 3)
             Time-optimised routing (urgency ≥ 4)
│
▼
[Section 3]  MILP Model (PuLP)
             Objective: Maximise net profit
             Variables: Procurement, Production, Shipment, Staff, Shortage
             Constraints: Supplier caps, Factory caps, BOM, Labour,
                          Warehouse, Demand, CO2 (500k kg hard cap)
│
▼
[Section 4]  Scenario Analysis (5 scenarios)
│
▼
[Sections 5–6]  Shadow Prices + Sensitivity
                Fuel shocks · Penalty shocks · Absenteeism shocks
│
▼
[Section 6D]  Contingency Plans
              Capella disruption · Demand surge · Buffer stock
│
▼
[Section 7]   Monte Carlo (500 runs × 2)
              Demand variance · Supply variance · Absenteeism
│
▼
[Sections 8–10]  Results: Fulfillment rates, Cost breakdown, CO2 metrics
│
▼
[Section 11]  14 Visualisations
│
▼
[Section 12]  Two-Stage Stochastic Programming
              5 demand scenarios · Stage-1/Stage-2 decisions · VSS
```

---

## 🧠 Modelling Techniques

| Technique | Implementation |
|---|---|
| **MILP** | PuLP `LpProblem` — maximise net profit across full supply chain |
| **Dijkstra's Algorithm** | NetworkX — cost-weighted and time-weighted graphs separately |
| **Two-Stage Stochastic Programming** | 5 demand scenarios, Stage-1 (procurement/staff/activation) + Stage-2 recourse (production/shipment/shortage) |
| **Monte Carlo Simulation** | 500+ iterations — demand variance + supply variance + absenteeism simultaneously |
| **Sensitivity Analysis** | Fuel cost (±10%, +20%), penalty multipliers (×1.5, ×2.0), absenteeism rate shocks |
| **Contingency Analysis** | Capella supplier failure, Bordeaux/Lyon shutdown, NY+Shanghai demand surge |

---

## 🏭 Data Summary (Tables 1–9 from Brief)

| Table | Description | Key Values |
|---|---|---|
| Table 1 | Suppliers | AromaVita (€42/kg), LuxeCap (€8/kg), EthanolPro (€3/kg), Capella (€60/kg) |
| Table 2 | Factories | Grasse 80k, Lyon 70k, Bordeaux 90k units/month |
| Table 3 | Workforce | Bottling 40 staff, Packaging 35, QC 25 (absenteeism 5–10%) |
| Table 4 | Demand | Paris 55k, Milan 35k, NY 65k (urgency 5), Tokyo 40k, Dubai 30k, Shanghai 50k |
| Table 5 | Warehouses | Paris 100k units, Marseille 120k units |
| Table 6 | Transport | Air €0.08/km, Sea €0.03/km, Rail €0.04/km, Road €0.05/km |
| Table 7A/B | Distance/Time | Paris WH → NY: 5,800 km / 6.4 hrs (air) |
| Table 8 | BOM | Product A: 5g oils, 50ml ethanol, 1 bottle, 2g extracts |
| Table 9 | Network Edges | 25-edge multimodal network via CDG, FRA, DXB, SIN hubs |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Khushi-Dhargawe/Global-Supply-Chain-Optimisation.git
cd Global-Supply-Chain-Optimisation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (~3–5 minutes for MC + SP)
python Supply_Chain_Optimisation_Groupe_Elegance.py
```

All 14 charts are saved automatically to the working directory.

> **Note:** The CBC solver (bundled with PuLP) is used. No additional solver installation required.

---

## 📊 Key Results by Section

| Section | Output | Business Action |
|---|---|---|
| Baseline scenario | Net loss €-1.7M (Table 3 exact staff limits) | Benchmark reference point |
| Optimal (no CO2 cap) | Net profit €18.3M | Shows true cost of sustainability cap — only €0.9M gap |
| **Optimal + CO2 cap** | **Net profit €17.4M** ← primary scenario | Primary operating scenario | 
| Bordeaux shutdown | Profit drops 0.2% (Lyon absorbs capacity) | Lyon absorbs capacity — low risk scenario |
| Capella disruption | Profit drops 109% (extracts are critical) | Dual-source rare extracts urgently |
| NY+Shanghai surge | €624k additional shortage penalties | Pre-position buffer stock at Paris warehouse |
| Monte Carlo VaR (5%) | €7.2M guaranteed floor (95% confidence) | Board-level risk planning figure |
| VSS (Stochastic SP) | ~€19M benefit of stochastic over deterministic planning | Justifies stochastic over deterministic planning |

---

## 🌱 Sustainability Metrics

| Metric | Value |
|---|---|
| CO₂ cap utilisation | 100% (binding constraint) |
| Best factory (CO₂/unit) | Bordeaux — 0.3 kg/unit |
| Worst factory (CO₂/unit) | Lyon — 0.5 kg/unit |
| Transport CO₂ share | ~78% of total emissions |
| SDG alignment | SDG 11 (urban freight), SDG 13 (climate action) |

**Recommendation:** Concentrate production at Bordeaux (1.1 kWh/unit, 0.3 kg CO₂/unit) and use sea freight from Marseille for non-urgent Asian markets to minimise emissions intensity per unit.

---

## What I'd Do With More Data
With real freight rate data I'd replace fixed transport costs with dynamic pricing models. With actual factory energy bills I'd extend the CO₂ model to include Scope 2 emissions and validate against Groupe Élégance's actual sustainability targets.

---

## 💼 Business Recommendations

1. **Concentrate production at Bordeaux** — lowest CO₂/unit (0.3 kg) and highest capacity (90,000 units); scale down Grasse and Lyon where possible.
2. **Dual-source rare extracts** — Capella disruption causes 109% profit drop; qualify a backup supplier immediately to eliminate this single point of failure.
3. **Sea freight for non-urgent Asian markets** — transport is 77.9% of total CO₂; switching Shanghai/Tokyo non-urgent orders from air to sea freight is the biggest emissions lever available.
4. **Build buffer stock at Paris warehouse** — pre-position inventory for NY (urgency 5, €8/unit penalty) and Tokyo (urgency 4) to absorb demand surges.
5. **Lock procurement and staffing decisions early** — Two-Stage SP shows Stage 1 decisions (€0.05M) are vastly cheaper than Stage 2 recourse costs (€1.63M), early commitment pays off.

---

## 📁 Related Projects
| # | Project | Skills |
|---|---|---|
| 1 | [Customer Shopping Behaviour Analysis](https://github.com/Khushi-Dhargawe/Customer-Shopping-Behaviour-Analysis) | Python · SQL · Power BI |
| 2 | [Ireland Housing Affordability Dashboard](https://github.com/Khushi-Dhargawe/Ireland-Housing-Affordability-Dashboard) | Python · Power BI |
| 3 | [Zepto Retail Analytics](https://github.com/Khushi-Dhargawe/Zepto-Retail-Analytics) | Python · SQL · PostgreSQL |
| 4 | [Customer Churn Prediction](https://github.com/Khushi-Dhargawe/Customer-Churn-Prediction) | Python · ML · SHAP · LIME |
| **5** | **Global Supply Chain Optimisation ← You are here** | **Python · MILP · Monte Carlo** |

---

## 👩‍💻 Author

**Khushi Dhargawe**  
MSc Business Analytics — University College Cork (UCC)  
BE Artificial Intelligence & Machine Learning (Hons. Cybersecurity) — Mumbai University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/khushi-dhargawe)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/Khushi-Dhargawe)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
