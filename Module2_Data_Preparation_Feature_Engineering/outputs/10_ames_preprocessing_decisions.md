# Preprocessing Decision Log: Ames Housing (1,460 sales) -> SalePrice

Training rows: 1168 (test sealed: 292). Final in-scope test RMSLE: 0.1266.

## 1. [parsing] all
- **Decision:** read_csv(keep_default_na=False, na_values=[''])
- **Why:** The string 'None' is a legitimate category in this data dictionary
- **Alternatives considered:** pandas defaults
- **Evidence:** MasVnrType: 872 'missing' with defaults vs 8 truly empty
- **Rows affected:** 864

## 2. [missing values] Alley, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature
- **Decision:** SimpleImputer(strategy='constant', fill_value='None')
- **Why:** Missing means 'feature absent' (no pool / garage / basement / fireplace), proven by the matching area/count column being 0
- **Alternatives considered:** mode imputation (invents pools)
- **Evidence:** PoolQC missing 1162 = PoolArea==0 1162

## 3. [missing values] LotFrontage
- **Decision:** median imputation + missing indicator
- **Why:** Genuinely unknown (no structural explanation); skewed -> median
- **Alternatives considered:** mean; neighbourhood median
- **Evidence:** 18.6% missing in train

## 4. [missing values] MasVnrArea
- **Decision:** fill 0 (with MasVnrType 'None')
- **Why:** 8 rows lack both veneer type and area; 'no veneer' is the most common state
- **Alternatives considered:** median

## 5. [feature types] MSSubClass, MoSold
- **Decision:** cast to string -> one-hot
- **Why:** Integer CODES, not quantities
- **Alternatives considered:** treat as numeric (fake order)

## 6. [feature selection] Id, Utilities, GarageYrBlt
- **Decision:** drop
- **Why:** identifier / near-constant / redundant with YearBuilt

## 7. [encoding] ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, GarageCond, PoolQC, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageFinish, Functional, LotShape, PavedDrive, CentralAir
- **Decision:** OrdinalEncoder with explicit order (None < Po < Fa < TA < Gd < Ex)
- **Why:** Quality scales are ORDERED; one-hot would discard the order and add ~60 columns
- **Alternatives considered:** one-hot (tested in ablation)

## 8. [target transform] SalePrice
- **Decision:** log1p via TransformedTargetRegressor (inverse expm1)
- **Why:** Right-skewed, multiplicative effects; evaluation metric is RMSLE
- **Alternatives considered:** raw target
- **Evidence:** skew 1.74 -> 0.12

## 9. [outliers / scope] GrLivArea, SalePrice
- **Decision:** exclude GrLivArea>4000 & SalePrice<300k from training and evaluation; flag such inputs at serving
- **Why:** Partial sales of unfinished homes (documented by the dataset author) - out of product scope
- **Alternatives considered:** keep; winsorize; drop all >4000 sq ft
- **Evidence:** 2 training rows

## 10. [scaling] MiscVal, PoolArea, LotArea, 3SsnPorch, LowQualFinSF, BsmtFinSF2, ScreenPorch, EnclosedPorch, LotFrontage, OpenPorchSF, MasVnrArea, BsmtFinSF1, TotalBsmtSF, WoodDeckSF, GrLivArea, 1stFlrSF, BsmtUnfSF, 2ndFlrSF, TotalSF, TotalPorchSF
- **Decision:** log1p, then StandardScaler
- **Why:** |skew| > 0.75 on train; areas are multiplicative
- **Alternatives considered:** StandardScaler only; PowerTransformer
- **Evidence:** 20 columns

## 11. [feature engineering] TotalSF, TotalBath, HouseAge, YearsSinceRemod, TotalPorchSF
- **Decision:** add TotalSF, TotalBath, HouseAge, YearsSinceRemod, TotalPorchSF
- **Why:** Appraiser logic: total space, total baths, age at sale, outdoor space
- **Alternatives considered:** let the model infer them
- **Evidence:** TotalSF is #2 by mutual information with log(price)

## 12. [ablation evidence] log target
- **Decision:** variant WORSE -> decision earns its place
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** +0.1715 RMSLE (noise band +/- 0.1912)

## 13. [ablation evidence] log1p skewed features
- **Decision:** within noise -> keep the simpler option
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** +0.0012 RMSLE (noise band +/- 0.0055)

## 14. [ablation evidence] domain features
- **Decision:** variant WORSE -> decision earns its place
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** +0.0014 RMSLE (noise band +/- 0.0011)

## 15. [ablation evidence] ordinal quality (one-hot instead)
- **Decision:** variant BETTER -> REVISIT the decision
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** -0.0028 RMSLE (noise band +/- 0.0011)

## 16. [ablation evidence] min_frequency (all rare levels kept)
- **Decision:** within noise -> keep the simpler option
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** -0.0001 RMSLE (noise band +/- 0.0004)

## 17. [ablation evidence] mean instead of median imputation
- **Decision:** within noise -> keep the simpler option
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** +0.0000 RMSLE (noise band +/- 0.0000)

## 18. [ablation evidence] QualxSF interaction (MI's #1 feature)
- **Decision:** variant BETTER -> REVISIT the decision
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** -0.0021 RMSLE (noise band +/- 0.0014)

## 19. [ablation evidence] outlier removal (train on partial sales too)
- **Decision:** variant WORSE -> decision earns its place
- **Why:** 5-fold paired CV: RMSLE of the variant minus the full pipeline
- **Evidence:** +0.0045 RMSLE (noise band +/- 0.0024)

## 20. [serving] all input fields
- **Decision:** fill unprovided fields with TRAIN mode/median before the pipeline; refuse GrLivArea > 4000
- **Why:** In a form, missing = 'not provided', not 'feature absent'
- **Alternatives considered:** leave NaN (structural 'None' -> severely underprices)
- **Evidence:** listing priced $100,217 with NaNs vs $273,640 with defaults
