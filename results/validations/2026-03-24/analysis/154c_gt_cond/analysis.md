# 154c: GT Conditionality Deep Investigation Results

## Key Finding: S3 turb/calm > 1.15 is UNREACHABLE on the test split

### Evidence
- Train turb/calm ratio (cross-window std): 1.492
- Test turb/calm ratio (cross-window std): 1.194 (INVERTED)
- Test bootstrap 95% CI: [1.128, 1.258]

### Why the inversion?
- Turb/calm std of levels: 1.122
- Future VoV ratio: 1.314
- Mean reversion: turb hist ends high, future IV drops
- Test period is a specific market regime (possibly low-vol grind after a spike)

### Implications for Model Development
1. Even a PERFECT model cannot pass S3 on this test split
2. The 1.15 threshold was derived from train data where the relationship holds
3. S3 is testing for a property that does NOT exist in the test data
4. Any model that passes S3 on test is either:
   a. Getting lucky with noise, or
   b. Overfitting to an artifact

### Recommendations
1. Re-evaluate S3 with the actual GT ratio as the target
2. Or use cross-validation across time periods instead of a fixed test split
3. Or test conditionality via a different metric (e.g., MAE reduction, which
   doesn't depend on the turb>calm assumption)
