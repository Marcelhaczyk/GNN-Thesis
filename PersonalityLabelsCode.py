import pandas as pd

# Read data from txt file
df = pd.read_csv('C:\\Users\marce\OneDrive\Pulpit\Praca magisterska\GNN\personality_labels.txt', sep=' ', header=None)
df.columns = ['ID', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']

participants_labels=[]

# Function to reverse score
def reverse_score(score):
    return 11 - score

# Calculate scores
df['Extraversion'] = df['Q1'].apply(reverse_score) + df['Q6']
df['Agreeableness'] = df['Q2'] + df['Q7'].apply(reverse_score)
df['Conscientiousness'] = df['Q3'].apply(reverse_score) + df['Q8']
df['Neuroticism'] = df['Q4'].apply(reverse_score) + df['Q9']
df['Openness'] = df['Q5'].apply(reverse_score) + df['Q10']

# Calculate percentage scores
df['Extraversion_pct'] = (df['Extraversion'] / 20 * 100).round(2)
df['Agreeableness_pct'] = (df['Agreeableness'] / 20 * 100).round(2)
df['Conscientiousness_pct'] = (df['Conscientiousness'] / 20 * 100).round(2)
df['Neuroticism_pct'] = (df['Neuroticism'] / 20 * 100).round(2)
df['Openness_pct'] = (df['Openness'] / 20 * 100).round(2)

# Print results
for index, row in df.iterrows():
    print(row['ID'])
    print("Extraversion:",row['Extraversion'],"(",row['Extraversion_pct'], "%)")
    print("Agreeableness:",row['Agreeableness'],"(",row['Agreeableness_pct'], "%)")
    print("Conscientiousness:",row['Conscientiousness'],"(",row['Conscientiousness_pct'], "%)")
    print("Neuroticism:",row['Neuroticism'],"(",row['Neuroticism_pct'], "%)")
    print("Openness:",row['Openness'],"(",row['Openness_pct'], "%)")
    print("\n")

    # Append scores to the list
    participants_labels.append([row['Extraversion'], row['Agreeableness'], row['Conscientiousness'], row['Neuroticism'], row['Openness']])

# Print participants_labels
print("participants_labels=", participants_labels)