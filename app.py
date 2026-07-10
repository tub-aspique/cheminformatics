import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Solubility Predictor", page_icon="🧪", layout="centered")

st.title("🧪 Molecular Solubility Predictor")
st.markdown("Enter a SMILES string to predict solubility in water and organic solvents.")

# ── Fonctions ─────────────────────────────────────────────────
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
    # Modèle eau
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    df = pd.read_csv(url)
    descripteurs = pd.DataFrame(df['smiles'].apply(calculer_descripteurs).tolist())
    fp_matrix = np.array([calculer_fingerprint(s) for s in df['smiles']])
    X = np.hstack([fp_matrix, descripteurs.values])
    y = df['measured log solubility in mols per litre'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_water = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_water.fit(X_train, y_train)

    # Modèles solvants organiques - RDKit + fingerprints uniquement
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
    
    return model_water, modeles_solvants

# ── Chargement des modèles ────────────────────────────────────
with st.spinner("Loading models... (first run may take a minute)"):
    model_water, modeles_solvants = charger_modeles()

# ── Interface ─────────────────────────────────────────────────
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
            
            # Prédiction eau
            desc = calculer_descripteurs(smiles_input)
            fp = calculer_fingerprint(smiles_input)
            X_mol = np.hstack([fp, list(desc.values())]).reshape(1, -1)
            logs_water = model_water.predict(X_mol)[0]
            
            # Résultats
            resultats = [{'Solvent': 'Water', 'Predicted logS': round(logs_water, 3)}]
            

            # Prédictions solvants organiques
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

            # Affichage eau
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