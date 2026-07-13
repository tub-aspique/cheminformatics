import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Solubility Predictor", page_icon="🧪", layout="centered")
st.title("🧪 Molecular Solubility Predictor")
st.markdown("Enter a SMILES string to predict solubility in water and organic solvents.")

def calculer_descripteurs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'LogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol)
    }

def calculer_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp)

@st.cache_resource
def charger_modeles():
    # Modèle eau - AqSolDB\ESOL
    df_aqsol = pd.read_csv('data/curated-solubility-dataset.csv')
    df_aqsol_clean = df_aqsol[['SMILES', 'Solubility']].rename(
        columns={'SMILES': 'smiles', 'Solubility': 'logS'})
    
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    df_esol = pd.read_csv(url)
    smiles_esol = set(df_esol['smiles'].tolist())
    df_aqsol_no_esol = df_aqsol_clean[~df_aqsol_clean['smiles'].isin(smiles_esol)]
    
    desc_list = df_aqsol_no_esol['smiles'].apply(calculer_descripteurs).tolist()
    valid_mask = [d is not None for d in desc_list]
    df_valid = df_aqsol_no_esol[valid_mask].reset_index(drop=True)
    desc_valid = pd.DataFrame([d for d in desc_list if d is not None])
    fp_valid = np.array([calculer_fingerprint(s) for s in df_valid['smiles']])
    
    X_water = np.hstack([fp_valid, desc_valid.values])
    y_water = df_valid['logS'].values
    model_water = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_water.fit(X_water, y_water)

    # Stocker les fingerprints pour le calcul de similarité Tanimoto
    smiles_train = df_valid['smiles']

    # Modèles solvants organiques
    modeles_solvants = {}
    for solvant, fichier in zip(['Ethanol', 'Benzene', 'Acetone'],
                                 ['ethanol_solubility_data.csv',
                                  'benzene_solubility_data.csv',
                                  'acetone_solubility_data.csv']):
        df_s = pd.read_csv(f'data/Solubility Data/{fichier}')
        desc_s = pd.DataFrame(df_s['SMILES'].apply(calculer_descripteurs).tolist())
        fp_s = np.array([calculer_fingerprint(s) for s in df_s['SMILES']])
        X_s = np.hstack([fp_s, desc_s.values])
        y_s = df_s['LogS'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_tr, y_tr)
        modeles_solvants[solvant] = model
    
    return model_water, modeles_solvants, smiles_train

with st.spinner("Loading models... (first run may take a minute)"):
    model_water, modeles_solvants, smiles_train = charger_modeles()


def tanimoto_max(smiles_query, smiles_database):
    mol_q = Chem.MolFromSmiles(smiles_query)
    fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, radius=2, nBits=2048)
    
    max_sim = 0.0
    # Échantillonner 1000 molécules aléatoires pour la vitesse
    sample = smiles_database.sample(min(1000, len(smiles_database)), random_state=42)
    for smi in sample:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sim = TanimotoSimilarity(fp_q, fp)
        if sim > max_sim:
            max_sim = sim
    return max_sim

smiles_input = st.text_input("SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)")

if st.button("Predict Solubility"):
    if not smiles_input:
        st.warning("Please enter a SMILES string.")
    else:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES. Please check your input.")
        else:
            st.success("Valid molecule detected!")
            
            desc = calculer_descripteurs(smiles_input)
            fp = calculer_fingerprint(smiles_input)
            X_mol = np.hstack([fp, list(desc.values())]).reshape(1, -1)
            
            # Similarité Tanimoto
            sim = tanimoto_max(smiles_input, smiles_train)
            if sim > 0.7:
                st.success(f"✅ Applicability domain: max Tanimoto similarity = {sim:.2f} — reliable prediction")
            elif sim > 0.4:
                st.warning(f"⚠️ Applicability domain: max Tanimoto similarity = {sim:.2f} — use with caution")
            else:
                st.error(f"❌ Applicability domain: max Tanimoto similarity = {sim:.2f} — molecule outside training domain")
            
            logs_water = model_water.predict(X_mol)[0]

            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Water logS", f"{logs_water:.3f}")
                if logs_water > -1:
                    st.success("Highly soluble in water")
                elif logs_water > -3:
                    st.warning("Moderately soluble in water")
                else:
                    st.error("Poorly soluble in water")
            with col2:
                st.markdown("**Molecular descriptors:**")
                st.dataframe(pd.DataFrame([desc]).T.rename(columns={0: 'Value'}))

            st.subheader("Organic Solvents")
            cols = st.columns(3)
            couleurs_solvants = {'Ethanol': '🟦', 'Benzene': '🟧', 'Acetone': '🟩'}
            for col, (solvant, model_s) in zip(cols, modeles_solvants.items()):
                logs_s = model_s.predict(X_mol)[0]
                with col:
                    st.metric(f"{couleurs_solvants[solvant]} {solvant} logS", f"{logs_s:.3f}")
                    if logs_s > -1:
                        st.success("Highly soluble")
                    elif logs_s > -2:
                        st.warning("Moderately soluble")
                    else:
                        st.error("Poorly soluble")