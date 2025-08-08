import pandas as pd
import seaborn as sns

# Original data
w8a8_7b = [1.207099423, 1.074974188, 1.03954899, 1.129713408, 0.96702853, 1.006905049]
w4a16_7b = [1.050884814, 0.9619349843, 0.9638136294, 1.025518866, 0.9076807015, 0.9562748442]
fp8_7b = [0.9818187626, 0.9561462778, 0.9591036977, 1.162057882, 1.050995438, 0.9903690938]
w8a8_14b = [1.354663551, 1.401041273, 1.350043821, 1.088317075, 1.075441334, 1.148100721]
w4a16_14b = [0.9959155738, 1.004906984, 1.088967144, 1.226019732, 1.338447995, 1.520007512]
fp8_14b = [1.087186046, 1.088865529, 1.184801954]
w8a8_32b = [1.75372741, 1.735281679, 1.38678965, 1.658840516, 1.748954372, 1.361242322]
w4a16_32b = [1.38262824, 1.455188638, 1.377282303, 1.327935588, 1.489729372, 1.322800535]
fp8_32b = [1.448393948, 1.599269673, 1.32839669]

def create_dataframe():
    """
    Convert the data lists into a pandas DataFrame with model_size and quantization_scheme columns.
    """
    # Create a list to store all rows
    rows = []
    
    # Define the data mapping
    data_mapping = {
        'w8a8_7b': {'data': w8a8_7b, 'model_size': 7, 'quantization_scheme': 'w8a8'},
        'w4a16_7b': {'data': w4a16_7b, 'model_size': 7, 'quantization_scheme': 'w4a16'},
        'fp8_7b': {'data': fp8_7b, 'model_size': 7, 'quantization_scheme': 'fp8'},
        'w8a8_14b': {'data': w8a8_14b, 'model_size': 14, 'quantization_scheme': 'w8a8'},
        'w4a16_14b': {'data': w4a16_14b, 'model_size': 14, 'quantization_scheme': 'w4a16'},
        'fp8_14b': {'data': fp8_14b, 'model_size': 14, 'quantization_scheme': 'fp8'},
        'w8a8_32b': {'data': w8a8_32b, 'model_size': 32, 'quantization_scheme': 'w8a8'},
        'w4a16_32b': {'data': w4a16_32b, 'model_size': 32, 'quantization_scheme': 'w4a16'},
        'fp8_32b': {'data': fp8_32b, 'model_size': 32, 'quantization_scheme': 'fp8'}
    }
    
    # Process each dataset
    for key, info in data_mapping.items():
        for value in info['data']:
            rows.append({
                'value': value,
                'model_size': info['model_size'],
                'quantization_scheme': info['quantization_scheme']
            })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df

# Create the DataFrame
df = create_dataframe()

sns.catplot(
    data=df, x="model_size", y="value", hue="quantization_scheme",
    zorder=1
)