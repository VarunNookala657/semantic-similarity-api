import requests
import pandas as pd
import time

# Load the dataset (Assuming it's a CSV file with columns 'text1' and 'text2')
df = pd.read_csv("DataNeuron_Text_Similarity.csv")  # Replace with actual dataset path

# API URL (Change this when deployed)
API_URL = "http://127.0.0.1:5001/similarity"  # Local testing

results = []  # Store results

# Loop through dataset
for index, row in df.iterrows():
    data = {
        "text1": row["text1"],
        "text2": row["text2"]
    }
    
    try:
        response = requests.post(API_URL, json=data)
        
        if response.status_code == 200:
            result = response.json()
            similarity_score = result.get("similarity score", "N/A")
            results.append([row["text1"], row["text2"], similarity_score])
        else:
            print(f"Error at row {index}: {response.text}")

    except Exception as e:
        print(f"Exception at row {index}: {str(e)}")
    
    time.sleep(0.1)  # Avoid overloading the API

# Save results to CSV
output_df = pd.DataFrame(results, columns=["text1", "text2", "similarity_score"])
output_df.to_csv("api_results.csv", index=False)

print("✅ API testing completed. Results saved in 'api_results.csv'.")
