import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS

def compute_mcs(df):
    mols = [Chem.MolFromSmiles(smiles) for smiles in df.SMILES]
    print(len(mols))
    for i in range(len(mols)):
        for m in (mols[:i] + mols[i+1:]):
            print(m)
            mcs = rdFMCS.FindMCS([mols[i], m])
            print(mcs.numAtoms)

def
    # fp1 = AllChem.GetMorganFingerprint(mols[0], radius=2)  # ECFP4-like
    # fp2 = AllChem.GetMorganFingerprint(mols[1], radius=2)
    # similarity = DiceSimilarity(fp1, fp2)  # Jaccard similarity
    # print("Tanimoto:", similarity)

if __name__ == '__main__':
    df = pd.read_csv('../CSV/Data/mol_length_6_test.csv')
    print(df)
    # compute_mcs(df[:5])
    smiles = ["CCCC[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C(=O)CN(CCOC)C(=O)[C@H](Cc2ccc3ccccc3c2)NC(=O)CN(Cc2ccc3c(c2)OCO3)C(=O)[C@H]2CCCN2C1=O",
              "CCCC[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C(=O)CN(CCCN2CCCC2=O)C(=O)[C@H](C)NC(=O)CN(Cc2ccc3c(c2)OCO3)C(=O)[C@H]2CCCN2C1=O"]
    mols = [Chem.MolFromSmiles(i) for i in smiles]

    params = rdFMCS.MCSParameters()
    # params.AtomTyper = CompareElementsOutsideRings()
    # params.BondTyper = CompareOrderOutsideRings()
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    mcs = rdFMCS.FindMCS(mols)
    print(mcs.numAtoms)  # Size of MCS
    similarity = mcs.numAtoms / max(mols[0].GetNumAtoms(), mols[1].GetNumAtoms())
    print("MCS Similarity:", similarity)

    # from rdkit.Chem import AllChem
    # from rdkit.DataStructs import DiceSimilarity, FingerprintSimilarity  # Renamed to FingerprintSimilarity
    #
    # fp1 = AllChem.GetMorganFingerprint(mols[0], radius=2)  # ECFP4-like
    # fp2 = AllChem.GetMorganFingerprint(mols[1], radius=2)
    # similarity = DiceSimilarity(fp1, fp2)  # Jaccard similarity
    # print("Tanimoto:", similarity)
