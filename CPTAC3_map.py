import os
import pandas as pd
from tqdm import tqdm
pd.options.mode.chained_assignment = None

df_new_sample = pd.read_csv('CPTAC-3&DKFZ/gene_dict_sample_new.csv')
geneid_dict = df_new_sample.set_index('GeneID')['ReplacementGeneID'].to_dict()
symbol_dict = df_new_sample.set_index('GeneID')['OfficialSymbol'].to_dict()
def remove_none_values(dictionary):
    # Create a new dictionary without None values
    new_dict = {str(key): value for key, value in dictionary.items() if not isinstance(value, float)}
    return new_dict
def remove_none_values_gene(dictionary):
    # Create a new dictionary without None values
    new_dict = {str(key): str(int(value)) for key, value in dictionary.items() if value > 0}
    return new_dict

def get_value_from_dict(dictionary, key):
    # Get the value from the dictionary if the key exists
    if key in dictionary:
        return dictionary[key]

    # Return the original value if the key is not in the dictionary
    return key

new_geneid_dict = remove_none_values_gene(geneid_dict)
new_symbol_dict = remove_none_values(symbol_dict)

CPTAC3_GeneExp_pth = 'CPTAC-3&DKFZ\GeneExp'
CPTAC3_GeneExp_save_pth = 'CPTAC-3&DKFZ\MatchedGeneExp'
files = os.listdir(CPTAC3_GeneExp_pth)

# Read the mapping DataFrame from CSV
mapping_df = pd.read_csv('CPTAC-3&DKFZ/gene_dict_symbol_new.csv')

# Create a dictionary mapping from the mapping DataFrame using '_id' as the key and 'MappedValue' as the value
mapping_dict = mapping_df.set_index('query')['_id'].to_dict()

header_df = pd.read_csv('CPTAC-3&DKFZ/gen_header.csv')
gen_header_get = header_df.values.tolist()
df_convert = pd.DataFrame(columns=gen_header_get)
match_record = []

for file in tqdm(files, desc='Processing files', unit='file'):
    file_get = os.listdir(os.path.join(CPTAC3_GeneExp_pth, file))
    tsv_name = ''
    for tsv_file in file_get:
        if tsv_file.endswith('.tsv'):
            tsv_name = tsv_file
            break
    gen_df = pd.read_csv(os.path.join(CPTAC3_GeneExp_pth, file, tsv_name), sep='\t', header=1)
    header = gen_df['gene_id'].values.tolist()

    content = gen_df['fpkm_unstranded'].tolist()
    df_get = pd.DataFrame({'gene_id': header, 'fpkm_unstranded': content})
    filtered_df = df_get[~df_get['gene_id'].str.contains('PAR_Y')]
    filtered_df['gene_id_new'] = filtered_df['gene_id'].apply(lambda x: x.split('.')[0])
    filtered_df = filtered_df.dropna().reset_index()

    # Read the main DataFrame from CSV
    main_df = filtered_df

    # Create a new column in the main DataFrame by mapping the values from the mapping DataFrame
    main_df['MappedColumn'] = main_df['gene_id_new'].map(mapping_dict)

    # Print the updated DataFrame
    gen_mapping_dict = main_df.set_index('MappedColumn')['fpkm_unstranded'].to_dict()

    count_pos = 0
    count_nag = 0
    value_list = []
    for column_name, column_data in df_convert.iteritems():
        try:
            column_name_get = str(column_name)[2: -3]
            # print(column_name_get)
            id = str(column_name_get.split('|')[1])
            new_id = get_value_from_dict(new_geneid_dict, id)
            value = gen_mapping_dict[new_id]
            # print(column_name_get, value)
            df_convert[column_name] = value
            count_pos += 1
            value_list.append([value])
        except:
            # print(column_name_get, 'None')
            df_convert[column_name] = None
            value_list.append(['NA'])
            count_nag += 1

    # Print the updated DataFrame
    df_convert_get = pd.DataFrame(value_list).transpose()
    df_convert_get.columns = df_convert.columns
    df_convert_get.to_csv(os.path.join(CPTAC3_GeneExp_save_pth, tsv_name.replace('.tsv', '.txt')), index=False)
    match_record.append([tsv_name, count_pos, count_nag - 5])
    #print([tsv_name, count_pos, count_nag - 5])
df_match_record = pd.DataFrame(match_record)
df_match_record.columns = ['TSVFileName', 'Matched_Gene_Num', 'Unmatched_Gene_Num']
df_match_record.to_csv('Match_Record.csv')