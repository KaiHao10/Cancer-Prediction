import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from pygam import LogisticGAM, s
from pygam import f
from functools import reduce
from operator import add
from pygam import te

# --- Load data ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train = train['y']
X_train = train.drop(columns=['y'])
X_test = test.copy()

# --- Handle missing values ---
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_df = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test_df = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# # For X_train
# for col in ['X44', 'X45']:
#     if X_train[col].isna().sum() > 0:
#         mask = ~X_train[col].isna()
#         predictors = X_train.drop(columns=['X5', 'X44', 'X45'])
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(predictors[mask])
#         clf = LogisticRegression(max_iter=5000)
#         clf.fit(X_scaled, X_train.loc[mask, col].astype(int))
#         missing = X_train[col].isna()
#         X_missing_scaled = scaler.transform(predictors[missing])
#         X_train.loc[missing, col] = clf.predict(X_missing_scaled)
#         print(f"Filled {missing.sum()} missing values in {col} using logistic regression.")

# # For X_test
# for col in ['X44', 'X45']:
#     if X_test[col].isna().sum() > 0:
#         mask = ~X_test[col].isna()
#         predictors = X_test.drop(columns=['X5', 'X44', 'X45'])
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(predictors[mask])
#         clf = LogisticRegression(max_iter=5000)
#         clf.fit(X_scaled, X_test.loc[mask, col].astype(int))
#         missing = X_test[col].isna()
#         X_missing_scaled = scaler.transform(predictors[missing])
#         X_test.loc[missing, col] = clf.predict(X_missing_scaled)
#         print(f"Filled {missing.sum()} missing values in {col} using logistic regression.")

# # Final check for any remaining missing values
# remaining_missing = X_train.isna().sum().sum()
# if remaining_missing == 0:
#     print("All missing values successfully filled.")
# else:
#     print(f"{remaining_missing} missing values remain in X_train.")

# --- Handle outliers ---
def winsor_df(df, limits=(0.01, 0.01)):
    return pd.DataFrame({col: winsorize(df[col], limits=limits) for col in df.columns})

X_train_df = winsor_df(X_train_df)
X_test_df = winsor_df(X_test_df)

# # --- Standardize features ---
# scaler = StandardScaler()
# X_train_df[X_train_df.columns] = scaler.fit_transform(X_train_df)
# X_test_df[X_test_df.columns] = scaler.transform(X_test_df)

# --- Remove redundant features ---
# Drop the second Glu column because it's incredibly weird
X_train_df = X_train_df.drop(columns=['X5'])
X_test_df = X_test_df.drop(columns=['X5'])

# --- Log transformation ---
features = X_train_df.copy()
skewness = features.skew().sort_values(ascending=False)
skewed_features = skewness[(skewness > 1) | (skewness < -1)].index.tolist()
print(f"Applying log1p to {len(skewed_features)} skewed features:\n", skewed_features)
# Apply log1p transformation to those features
X_train_df[skewed_features] = np.log1p(X_train_df[skewed_features])
X_test_df[skewed_features] = np.log1p(X_test_df[skewed_features])

# # --- Create some interation terms ---
# corr_matrix = X_train_df.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# top_corr_pairs = upper.stack().sort_values(ascending=False).head(5)
# print("Top correlated feature pairs:\n", top_corr_pairs)

# # Create interaction terms for top 3 pairs
# top_pairs = top_corr_pairs.index[:3]
# for i, (feat1, feat2) in enumerate(top_pairs):
#     inter_col = f'inter_{feat1}_{feat2}'
#     X_train_df[inter_col] = X_train_df[feat1] * X_train_df[feat2]
#     X_test_df[inter_col] = X_test_df[feat1] * X_test_df[feat2]

# --- Create some interaction terms intuitively ---
interaction_pairs = [
     ('X6', 'X7'),   # Creatinine * Urea — kidney function
     ('X12', 'X13'), # Direct Bilirubin * Indirect Bilirubin — liver function
     ('X14', 'X15'), # Total Protein * Albumin — protein metabolism
     ('X16', 'X17'), # A/G Ratio * ALT — liver stress marker
     ('X19', 'X21'), # ALP * GGT — cholestasis
     ('X26', 'X27'), # WBC * Lymphocytes — immune response
     ('X32', 'X33'), # RBC * Hemoglobin — anemia evaluation
     ('X36', 'X37'), # MCH * MCHC — red blood cell indices
     ('X40', 'X41'), # MPV * PCT — platelet volume and percentage
     ('X19', 'X45'), # ALP * Fecal Transferrin
     ('X33', 'X44'), # Hemoglobin * Fecal Occult Blood
     ('X8', 'X44')  # CEA * Fecal Occult Blood
]
for feat1, feat2 in interaction_pairs:
     inter_col = f'inter_{feat1}_{feat2}'
     X_train_df[inter_col] = X_train_df[feat1] * X_train_df[feat2]
     X_test_df[inter_col] = X_test_df[feat1] * X_test_df[feat2]

# # --- Dimension reduction ---
# # Only used in the code testing phase to save time
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_df)
# X_test_scaled = scaler.transform(X_test_df)
# pca = PCA(n_components=0.95, random_state=42)
# X_train_df = pd.DataFrame(pca.fit_transform(X_train_scaled))
# X_test_df = pd.DataFrame(pca.transform(X_test_scaled))

# --- Train-validation split ---

X_tr, X_val, y_tr, y_val = train_test_split(X_train_df, y_train, test_size=0.2, stratify=y_train, random_state=42)
# Ensure consistent column order after all column modifications (especially interactions)
X_val = X_val[X_tr.columns]
X_test_df = X_test_df[X_tr.columns]

# Assert column alignment before model training
assert all(X_val.columns == X_tr.columns), "Mismatch in validation columns"
assert all(X_test_df.columns == X_tr.columns), "Mismatch in test columns"

# --- Model buliding ---
# model_l1 = LogisticRegressionCV(
#     penalty='l1',
#     solver='liblinear',
#     cv=5,
#     scoring='f1',
#     random_state=42,
#     max_iter=1000
# )
# model_l1.fit(X_tr, y_tr)
# print(X_train_df.columns.get_loc('X44'), X_train_df.columns.get_loc('X45'))
# print(X_train_df.columns.get_loc('X6'), X_train_df.columns.get_loc('X7'),
#       X_train_df.columns.get_loc('X33'), X_train_df.columns.get_loc('X44'), 
#       X_train_df.columns.get_loc('X19'), X_train_df.columns.get_loc('X45'))

print("Fitting Logistic GAM...")
# Dynamically construct terms for GAM based on column names
# Map column names to indices
col_to_idx = {col: idx for idx, col in enumerate(X_tr.columns)}

# Define indices by column name for categorical or excluded variables
exclude_cols = ['X44', 'X45']
categorical_cols = ['X43', 'X44']

# Construct the terms dynamically
terms = reduce(add, [s(i) for col, i in col_to_idx.items() if col not in exclude_cols])
for col in categorical_cols:
    terms += f(col_to_idx[col])

# --- Hyperparameter tuning for n_splines and lam ---
n_splines_options = [5, 10, 15]
lam_options = [0.1, 1, 10]
best_f1 = -1
best_model = None

for ns in n_splines_options:
    for lam_val in lam_options:
        model = LogisticGAM(terms, n_splines=ns, lam=lam_val).fit(X_tr.values, y_tr.values)
        val_probs = model.predict_proba(X_val.values)
        score = f1_score(y_val, val_probs > 0.5)  # Use fixed threshold for comparison
        if score > best_f1:
            best_f1 = score
            best_model = model
            best_params = (ns, lam_val)

print(f"Best n_splines: {best_params[0]}, Best lam: {best_params[1]}")
model_gam = best_model

# # --- Drop non-significant features based on p-value ---
# print("Filtering non-significant features (p > 0.2)...")
# # Manually extract term and p-values, EDoFs
# term_info = []
# for i, term in enumerate(model_gam.terms):
#     name = term.__repr__()
#     p_val = model_gam.statistics_['p_values'][i]
#     term_info.append({'term_idx': i, 'Feature Function': name, 'P > x': p_val})

# summary_df = pd.DataFrame(term_info)

# # Only smooth terms (s(i)) with p > 0.2
# smooth_terms = summary_df[summary_df['Feature Function'].str.startswith('s(')]
# drop_terms = smooth_terms[smooth_terms['P > x'] > 0.2]

# print("Dropping the following features due to high p-value (> 0.2):")
# print(drop_terms[['Feature Function', 'P > x']])

# # Extract column indices to drop
# drop_features = [X_tr.columns[int(name[2:-1])] for name in drop_terms['Feature Function']]

# # Drop from training, validation, and test sets
# X_tr = X_tr.drop(columns=drop_features)
# X_val = X_val.drop(columns=drop_features)
# X_test_df = X_test_df.drop(columns=drop_features)

# # Update col_to_idx and rebuild terms
# col_to_idx = {col: idx for idx, col in enumerate(X_tr.columns)}
# terms = reduce(add, [s(i) for col, i in col_to_idx.items() if col not in exclude_cols])
# for col in categorical_cols:
#     terms += f(col_to_idx[col])

# # Refit the model
# model_gam = LogisticGAM(terms, n_splines=best_params[0], lam=best_params[1]).fit(X_tr.values, y_tr.values)

# # --- Feature Importance ---
# model_gam.summary()

# influences = []
# for i in range(X_tr.shape[1]):
#     try:
#         XX = model_gam.generate_X_grid(term=i)
#         pd = model_gam.partial_dependence(term=i, X=XX)
#         influence_score = np.mean(np.abs(np.gradient(pd.flatten())))
#         influences.append((i, X_tr.columns[i], influence_score))
#     except:
#         continue  # skip if the term is not numeric or causes error

# # Sort and display top 10
# influences.sort(key=lambda x: x[2], reverse=True)
# print("Top 10 most influential features:")
# for idx, name, score in influences[:10]:
#     print(f"{name} (term {idx}): influence score = {score:.4f}")

# --- Threshold Optimization ---
probs = model_gam.predict_proba(X_val.values)
thresholds = np.linspace(0.2, 0.8, 21)
f1_scores = []
for t in tqdm(thresholds, desc="Evaluating thresholds"):
    score = f1_score(y_val, probs > t)
    f1_scores.append(score)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold: {best_threshold}")
print(f"Best F1 score: {max(f1_scores)}")

# Print threshold vs F1 score
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, marker='o')
plt.title("F1 Score vs. Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.grid(True)
plt.axvline(x=best_threshold, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("f1_vs_threshold.png")
plt.show()

# --- Final Predictions ---
# final_probs = model_l1.predict_proba(X_test_df)[:, 1]
final_probs = model_gam.predict_proba(X_test_df.values)
final_preds = (final_probs > best_threshold).astype(int)
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'y': final_preds
})
submission.to_csv('submission.csv', index=False)
