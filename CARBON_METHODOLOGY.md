# 碳排放核算方法学学术溯源

## 溯源总图（IPCC AR6 GWP-100）

```
IPCC 2006 Guidelines
  │
  ├─ Vol.2, Ch.2 (Stationary Combustion)
  │   ├─ Table 2.2 → 煤炭排放因子 (94.6 kgCO₂/GJ)
  │   ├─ Table 2.2 → 天然气排放因子 (56.1 kgCO₂/GJ)
  │   └─ Table 3.2.1 → 柴油排放因子 (2.68 kgCO₂/L)
  │
  ├─ Vol.2, Ch.3 (Mobile Combustion)
  │   └─ Table 3.2.1 → CH₄/N₂O per L diesel (0.0002/0.0001)
  │
  ├─ Vol.4, Ch.11 (N₂O from Managed Soils)
  │   ├─ Table 11.1 → N₂O EF (0.01 kg N₂O-N / kg N)
  │   └─ Eq. 11.1  → N₂O emissions = N_input × EF × 44/28
  │
  └─ IPCC AR6 (2021) → GWP-100 values
      ├─ CH₄ (fossil)    = 27.0  (AR6, Ch.7, Table 7.15)
      ├─ CH₄ (non-fossil) = 29.8  (AR6, Ch.7, Table 7.15)
      └─ N₂O             = 273   (AR6, Ch.7, Table 7.15)
```

## 逐因子溯源表

| 排放源 | 因子值 | IPCC出处 | 学术文献 |
|--------|--------|----------|----------|
| 氮肥N₂O排放 | 0.01 kg N₂O-N/kg N | IPCC 2006 Vol.4 Ch.11 Table 11.1 | Bouwman et al. (2002) Global Biogeochem. Cyc. 16:1058 |
| N₂O GWP | 273 | IPCC AR6 (2021) Ch.7 Table 7.15 | IPCC Sixth Assessment Report |
| CH₄ GWP (化石源) | 27 | IPCC AR6 (2021) Ch.7 Table 7.15 | IPCC Sixth Assessment Report |
| CH₄ GWP (非化石源) | 29.8 | IPCC AR6 (2021) Ch.7 Table 7.15 | IPCC Sixth Assessment Report |
| 柴油CO₂ | 2.68 kg/L | IPCC 2006 Vol.2 Ch.3 Table 3.2.1 | — (default EF for gas/diesel oil) |
| 柴油CH₄ | 0.0002 kg/L | IPCC 2006 Vol.2 Ch.3 Table 3.2.1 | — |
| 柴油N₂O | 0.0001 kg/L | IPCC 2006 Vol.2 Ch.3 Table 3.2.1 | — |
| 中国电网 | 0.5703 kg/kWh | 中国生态环境部 (2022) | — (2022年度中国电网排放因子) |
| 泰国电网 | 0.4892 kg/kWh | EGAT 2022 Annual Report | — |
| 越南电网 | 0.5238 kg/kWh | MOIT Decision 2726/QD-BCT | — |
| 蔗叶焚烧CO₂ | 1500 kg/ton | — | Andreae & Merlet (2001) GBC 15(4):955-966 |
| 蔗叶焚烧CH₄ | 2.3 kg/ton | — | França et al. (2012) Atm. Env. 62:247-255 |
| 蔗叶焚烧N₂O | 0.07 kg/ton | — | Silva et al. (2017) Sci. Total Env. 605:1195-1204 |
| 生物质替代煤炭 | -1800 kgCO₂/ton | — | Cherubini et al. (2009) Resour. Conserv. Recy. 53:434-447 |
| 土壤碳封存(免耕) | -500 kgCO₂/ha/yr | IPCC 2019 Refinement Vol.4 Ch.5 Table 5.5 | Lal (2004) Science 304:1623-1627 |
| 土壤碳封存(覆盖) | -300 kgCO₂/ha/yr | IPCC 2019 Refinement Vol.4 Ch.5 Section 5.3.2 | Poeplau & Don (2015) GCB 21:658-672 |

## 本项目碳核算公式链（AR6 GWP）

### 1. 种植环节（化肥N₂O）

```
N₂O_emissions = Fertilizer_N_kg × 0.01 × (44/28) × 273
              = Fertilizer_N_kg × 0.01 × 1.571 × 273
              = Fertilizer_N_kg × 4.289
```

出处：IPCC 2006 Vol.4 Ch.11 Eq. 11.1 + AR6 GWP-100 (N₂O=273)

### 2. 机械作业（柴油）

```
CO₂_diesel  = Diesel_L × 2.68
CH₄_diesel  = Diesel_L × 0.0002 × 27
N₂O_diesel  = Diesel_L × 0.0001 × 273
```

出处：IPCC 2006 Vol.2 Ch.3 Table 3.2.1 + AR6 GWP

### 3. 加工环节（电力）

```
CO₂_electricity = Electricity_kWh × Grid_Emission_Factor(country)
```

出处：各国政府官方发布（中国MEE 2022 / 泰国EGAT 2022 / 越南MOIT 2022）

### 4. 蔗叶焚烧（传统模式）

```
Total_CO₂e = Leaf_ton × (
    1500 × 1          (CO₂, 生物源=0)
  + 2.3  × 29.8       (CH₄, 非化石源GWP)
  + 0.07  × 273       (N₂O)
)
= Leaf_ton × 1587.65 kg CO₂e/ton
```

出处：Andreae & Merlet (2001); IPCC AR6 GWP-100

### 5. 生物质替代煤炭（最优模式）

```
Carbon_Reduction = Leaf_ton × 1800 × co_firing_ratio
```

出处：Cherubini et al. (2009); co_firing_ratio = 1.0(理论) / 0.3(行业标准)

### 6. 滤泥填埋CH₄

```
CH₄ = Mud_ton × 0.25 × 0.20 × 0.50 × 0.60 × (16/12) × 29.8
    = Mud_ton × 0.596 kg CO₂e/ton
```

出处：IPCC 2006 Vol.5 Ch.3 (Solid Waste Disposal), 一级降解有机碳(DOC)法简化

## 方法学分级（Tier体系）

| 环节 | Tier | 说明 |
|------|------|------|
| 化肥N₂O | **Tier 1** | IPCC默认排放因子 × 国家级活动数据 |
| 柴油燃烧 | **Tier 1** | IPCC默认排放因子 × 燃料消耗量 |
| 电力消耗 | **Tier 2** | 国家级电网排放因子（官方发布） |
| 蔗叶焚烧 | **Tier 2** | 文献实测排放因子（优于IPCC默认值） |
| 生物质替代 | **Tier 2** | 文献综述均值 + 掺烧比（行业标准） |

## AR5 → AR6 GWP变化与影响

| 温室气体 | AR5 (2014) | AR6 (2021) | 变化 | 本项目影响 |
|:---|:---:|:---:|:---:|:---|
| N₂O | 298 | 273 | -8.4% | 化肥N₂O排放估算降低8.4% |
| CH₄ (化石源) | 28 | 27 | -3.6% | 柴油CH₄排放估算降低3.6% |
| CH₄ (非化石源) | 28 | 29.8 | +6.4% | 甘蔗焚烧CH₄排放估算增加6.4% |

**净影响**：由于N₂O权重最大（化肥N₂O是主要排放源），AR6升级后总体碳排放估算**略有降低**，更符合最新科学共识。

## 与其他碳核算体系的对接

```
本项目 → 广西地方碳普惠方法学（建议纳入甘蔗副产物碳汇）
      → 全国碳市场 CEA（碳配额交易，2026-07-22收盘价 91.03元/吨）
      → 国际自愿碳市场 VCS/GS（约 50-150 元/吨，取决于方法学）
      → 来宾东糖全国首个糖业碳汇监测系统（实时数据对接潜力）
```

## 参考文献

1. IPCC. 2006 IPCC Guidelines for National Greenhouse Gas Inventories
2. IPCC. AR6 Climate Change 2021: The Physical Science Basis. Contribution of Working Group I
3. IPCC. 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories
4. Andreae MO, Merlet P. Emission of trace gases and aerosols from biomass burning. Global Biogeochem. Cyc., 2001, 15(4): 955-966
5. Cherubini F, et al. Energy- and greenhouse gas-based LCA of biofuel and bioenergy systems. Resour. Conserv. Recy., 2009, 53: 434-447
6. 中国生态环境部. 2022年度中国电网排放因子. 2022