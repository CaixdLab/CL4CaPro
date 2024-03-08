import pandas as pd
import mygene

mg = mygene.MyGeneInfo()

df_get = pd.read_csv('gene_id_get.csv')
# Example ENSEMBL IDs to convert
ensembl_ids = df_get.gene_id.values.tolist()

new_ens = []
for ensembl_id in ensembl_ids:
    new_ens.append(ensembl_id.split('.')[0])

geneSyms = mg.querymany(new_ens , scopes='ensembl.gene', fields='all', species='human')

df = pd.DataFrame.from_dict(geneSyms)
df.to_csv('gene_dict.csv')