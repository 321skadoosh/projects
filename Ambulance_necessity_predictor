import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import chi2_contingency

# Load datasets with selected columns
person_df = pd.read_csv('person.csv',
                        usecols=['ACCIDENT_NO', 'SEATING_POSITION', 'INJ_LEVEL', 'HELMET_BELT_WORN', 'AGE_GROUP'])
vehicle_df = pd.read_csv('filtered_vehicle.csv',
                         usecols=['ACCIDENT_NO', 'LEVEL_OF_DAMAGE'])
accident_df = pd.read_csv('accident.csv',
                          usecols=['ACCIDENT_NO', 'SPEED_ZONE'])
weather_df = pd.read_csv('atmospheric_cond.csv', usecols=['ACCIDENT_NO', 'ATMOSPH_COND'])

# Merge all datasets on ACCIDENT_NO
df1 = pd.merge(person_df, accident_df, on='ACCIDENT_NO')
df2 = pd.merge(df1, vehicle_df, on='ACCIDENT_NO')
final_df = pd.merge(df2, weather_df, on='ACCIDENT_NO', how='left')

key_vars = ['ACCIDENT_NO', 'INJ_LEVEL']

# Remove entries with missing ACCIDENT_NO or INJ_LEVEL
final_df = final_df.dropna(subset=key_vars)

# Handle 'Unknown' / 'NA' / 'NK' as valid category for now
def clean_category(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip().upper()
    if val_str in ['NA', 'N/A', 'NK', 'UNKNOWN', '']:
        return 'Unknown'
    return val


for col in ['SPEED_ZONE', 'SEATING_POSITION', 'HELMET_BELT_WORN', 'AGE_GROUP', 'LEVEL_OF_DAMAGE', 'ATMOSPH_COND']:
    final_df[col] = final_df[col].apply(clean_category)

# Preprocess LEVEL_OF_DAMAGE into fewer categories
def categorize_damage(damage_code):
    if damage_code in [1,6]:
        return 'Minor'
    elif damage_code in [2, 3]:
        return 'Moderate'
    elif damage_code in [4, 5]:
        return 'Major'
    elif damage_code == 9:
        return 'Unknown'
    else:
        return 'Unknown'

final_df['LEVEL_OF_DAMAGE'] = final_df['LEVEL_OF_DAMAGE'].apply(categorize_damage)

# Preprocess ATMOSPH_COND
def categorize_atmosphere(atmosph_code):
    if atmosph_code == 1:
        return 'Normal'
    elif atmosph_code == 9:
        return 'Unknown'
    else:
        return 'Abnormal'

final_df['ATMOSPHERE_CATEGORY'] = final_df['ATMOSPH_COND'].apply(categorize_atmosphere)

# Broaden AGE_GROUP using the provided function
def broaden_age_range(age_group):
    # in case age group is 'Unknown'
    if isinstance(age_group, str) and age_group[0].isalpha():
        return age_group

    # in case age_group is above 65
    if isinstance(age_group, str) and ('+' in age_group or (age_group.count('-') == 1 and int(age_group.split('-')[0]) >= 65)):
        return '65+'
    # finds starting and ending age of age group
    if isinstance(age_group, str) and '-' in age_group:
        start_age, end_age = map(int, age_group.split('-'))
        if 40 <= start_age and end_age <= 64:
            return '40-64'
        elif 26 <= start_age and end_age <= 39:
            return '26-39'
        elif 17 <= start_age and end_age <= 25:
            return '17-25'
        elif 0 <= start_age and end_age <= 16:
            return 'Under 16'
        # in case age group is between the new age ranges, for example '20-30'
        else:
            return age_group
    return 'Unknown'

final_df['AGE_GROUP'] = final_df['AGE_GROUP'].apply(broaden_age_range)

# Create AMBULANCE_NEEDED target variable
def ambulance_needed(injury_level):
    """
    Determines if an ambulance is likely needed based on the injury level.
    Ambulance is assumed to be needed for Fatal or Serious injuries.
    """
    if injury_level in [1, 2]:  # 1: Fatal, 2: Serious
        return 'Yes'
    else:
        return 'No'

final_df['AMBULANCE_NEEDED'] = final_df['INJ_LEVEL'].apply(ambulance_needed)


# Classify seatbelt usage
def classify_seatbelt(x):
    if x == 1:
        return 'Worn'
    elif x == 8:
        return 'Not_Worn'
    else:
        return 'Unknown'


final_df['SEATBELT_STATUS'] = final_df['HELMET_BELT_WORN'].apply(classify_seatbelt)


# Clean and categorize seating positions
def categorize_seating(pos):
    if pos == 'DR':
        return 'Driver'
    elif pos in ['LF', 'CF', 'PL']:
        return 'Front_Passenger'
    elif pos in ['RR', 'CR', 'LR', 'OR']:
        return 'Rear_Passenger'
    else:
        return 'Unknown'


final_df['SEATING_POSITION'] = final_df['SEATING_POSITION'].apply(categorize_seating)

# Categorize speed zones
def categorize_speed_zone(zone):
    if zone > 200:
        return 'Unknown'
    elif zone < 60:
        return 'Slow'
    elif 60 <= zone <= 80:
        return 'Medium'
    else:
        return 'Fast'

final_df['SPEED_ZONE'] = final_df['SPEED_ZONE'].apply(categorize_speed_zone)

# Define variables for analysis
cat_variables = ['SEATBELT_STATUS', 'SEATING_POSITION', 'SPEED_ZONE', 'AGE_GROUP', 'LEVEL_OF_DAMAGE', 'ATMOSPHERE_CATEGORY']
target = 'AMBULANCE_NEEDED'

# Cramér's V function (used later for effect size)
def cramers_v_stat(contingency_table):
    """Calculate Cramér's V statistic for correlation analysis"""
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# NMI Scores
nmi_scores = {}
for var in cat_variables:
    score = normalized_mutual_info_score(final_df[var], final_df[target], average_method='min')
    nmi_scores[var] = score

# Cramér's V (Effect Size)
cramers_v_scores = {}
for var in cat_variables:
    contingency = pd.crosstab(final_df[var], final_df[target])
    cv = cramers_v_stat(contingency)
    cramers_v_scores[var] = cv

# NMI Scores visualization
plt.figure(figsize=(12, 7))
pd.Series(nmi_scores).sort_values().plot(
    kind='barh',
    color='skyblue',
    title='Normalized Mutual Information with Ambulance Needed'
)
plt.xlabel("NMI Score")
plt.tight_layout()
plt.savefig('nmi_scores.png')

# Cramér's V visualization
plt.figure(figsize=(12, 7))
pd.Series(cramers_v_scores).sort_values().plot(
    kind='barh',
    color='skyblue',
    title='Categorical Variables Effect on Ambulance Needed (Cramér\'s V)'
)
plt.xlabel("Cramér's V Score")
plt.tight_layout()
plt.savefig('cramers_v_scores.png')

# Heatmap for conditional probabilities for top 3 variables based on correlation analysis
selected_vars = sorted(cat_variables, key=lambda x: cramers_v_scores.get(x, 0), reverse=True)[:3]
for var in selected_vars:
    plt.figure(figsize=(14, 7))
    cond_probs = pd.crosstab(final_df[var], final_df[target], normalize='index') * 100
    sns.heatmap(cond_probs, annot=True, cmap="YlGnBu", fmt='.1f', cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Effect of {var} on Ambulance Needed')
    plt.tight_layout()
    plt.savefig(f'{var}_ambulance_needed_heatmap.png')

# Split the data (70% training, 30% testing)
X = final_df[cat_variables]
y = final_df['AMBULANCE_NEEDED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Encode categorical features after splitting
X_train_encoded = pd.get_dummies(X_train, columns=cat_variables, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=cat_variables, drop_first=True)
train_cols = X_train_encoded.columns
test_cols = X_test_encoded.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_encoded[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_encoded[c] = 0

X_test_encoded = X_test_encoded[train_cols]

# Model 1 (Random Forest)
print("\nStarting Random Forest Classifier training...")
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced_subsample',
    random_state=42
)
model_rf.fit(X_train_encoded, y_train)

# Predict
y_pred_rf = model_rf.predict(X_test_encoded)

# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", round(accuracy_rf, 4))

# Classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=sorted(y.unique())))

# Confusion Matrix for Random Forest (matching Model 2's style)
cm_rf = confusion_matrix(y_test, y_pred_rf)
class_names_binary = sorted(y.unique())

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names_binary, yticklabels=class_names_binary)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Ambulance Needed Prediction (Random Forest)')
plt.tight_layout()
plt.savefig('confusion_matrix_random_forest.png')

# Normalized Confusion Matrix for Random Forest (matching Model 2's style)
cm_percent_rf = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent_rf, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=class_names_binary, yticklabels=class_names_binary)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix: Ambulance Needed Prediction (Random Forest)')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized_random_forest.png')

# feature importance calculation
importances = model_rf.feature_importances_
feature_names = X_train_encoded.columns

grouped_importance = {var: 0 for var in cat_variables}
for feature, importance in zip(feature_names, importances):
    for var in cat_variables:
        if feature.startswith(var + '_'):  # Adjust to match dummy column names
            grouped_importance[var] += importance
            break
        elif feature == var: # In case a category only has one level after dropping first
            grouped_importance[var] += importance
            break

# Convert to percentages
grouped_importance = {k: v * 100 for k, v in grouped_importance.items()}

# Sort the feature importance percentages, then plot
sorted_items = sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True)
vars_sorted, importances_sorted = zip(*sorted_items)

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances_sorted)), importances_sorted, align='center')
plt.yticks(range(len(importances_sorted)), vars_sorted)
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance (%)')
plt.title('Grouped Feature Importance (% of Total) - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance_random_forest.png')

# Model 2 (CatBoost)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Initialize and train CatBoost classifier
model_cb = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,  # Slightly high, not as robust but learns faster with lesser iterations
    depth=6,
    loss_function='Logloss',  # Good for predicting binary variables like ambulance needed
    random_seed=42,
    verbose=100,  # prints information on model's learning progress every 100 iterations
    thread_count=-1,
    cat_features=list(range(X_train_encoded.shape[1])),
    class_names=list(y_train.unique()),
    early_stopping_rounds=50,  # prevents model from overfitting
    class_weights=class_weight_dict
)

print("\nStarting CatBoost model training...")
model_cb.fit(
    X_train_encoded, y_train,
    eval_set=(X_test_encoded, y_test),
    plot=False
)

# Make predictions
y_pred_cb = model_cb.predict(X_test_encoded)
y_pred_proba_cb = model_cb.predict_proba(X_test_encoded)[:, 1]

# Evaluate the model
accuracy_cb = accuracy_score(y_test, y_pred_cb)
print(f"\nCatBoost Accuracy on the test set: {accuracy_cb:.4f}")
print("\nCatBoost Classification Report:")
print(classification_report(y_test, y_pred_cb, target_names=sorted(y.unique())))

# Confusion Matrix
cm_cb = confusion_matrix(y_test, y_pred_cb)
class_names_binary = sorted(y.unique())

plt.figure(figsize=(8,6))
sns.heatmap(cm_cb, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names_binary, yticklabels=class_names_binary)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Ambulance Needed Prediction (CatBoost)')
plt.tight_layout()
plt.savefig('confusion_matrix_catboost.png')

# Normalized Confusion Matrix
cm_normalized_cb = cm_cb.astype('float') / cm_cb.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized_cb, annot=True, fmt='.2f', cmap='Blues',
           xticklabels=class_names_binary, yticklabels=class_names_binary)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix: Ambulance Needed Prediction (CatBoost)')
plt.tight_layout()
plt.savefig('confusion_matrix_normalized_catboost.png')

# Calculate Feature Importance for CatBoost
feature_importance_cb = model_cb.get_feature_importance()
feature_importance_cb = feature_importance_cb / feature_importance_cb.sum() * 100

feature_names_encoded_cb = X_train_encoded.columns
grouped_importance_cb = {var: 0 for var in cat_variables}

for i, importance in enumerate(feature_importance_cb):
    encoded_feature_name = feature_names_encoded_cb[i]
    for var in cat_variables:
        if encoded_feature_name.startswith(var + '_') or encoded_feature_name == var:
            grouped_importance_cb[var] += importance
            break

# Sort these feature importances, then plot them
sorted_items_cb = sorted(grouped_importance_cb.items(), key=lambda x: x[1], reverse=True)
vars_sorted_cb, importances_sorted_cb = zip(*sorted_items_cb)

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances_sorted_cb)), importances_sorted_cb, align='center')
plt.yticks(range(len(importances_sorted_cb)), vars_sorted_cb)
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance (%)')
plt.title('Grouped Feature Importance (% of Total) - CatBoost')
plt.tight_layout()
plt.savefig('feature_importance_catboost_normalized.png')

plt.show()
