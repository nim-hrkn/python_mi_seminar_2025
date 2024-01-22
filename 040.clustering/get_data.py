import pandas as pd
import os


def load(data_name: str, root_dir=".."):
    """
    Load a specified dataset and return a DataFrame, a list of descriptor names, and the name of the target variable.

    This function reads a dataset based on the provided dataset name. It handles various predefined datasets and 
    loads them into a pandas DataFrame. Additionally, it identifies the descriptor names (features) and the 
    target variable name for each dataset.

    Parameters:
    data_name (str): The name of the dataset to load. Supported datasets are 'x5_sin', 'x5_sin_new', 'ReCo', 
                     'Carbon8', 'ZB_WZ_all', 'ZB_WZ_3', 'ZB_WZ_2', 'Carbon8_desc', 'Carbon8_desc_all', 
                     'Fe2', 'Fe2_new', and 'mono'.

    root_dir (str, optional): The root directory where the dataset files are located. Defaults to "..".

    Returns:
    tuple: A tuple containing three elements:
        - df (pandas.DataFrame): The DataFrame containing the dataset.
        - descriptor_names (list of str): A list of the names of the descriptor columns in the DataFrame.
        - target_name (str): The name of the target variable column in the DataFrame.

    Raises:
    RuntimeError: If the specified dataset name does not match any of the predefined datasets.

    Examples:
    >>> df, descriptors, target = load("x5_sin")
    >>> print(df.head())
    >>> print(descriptors)
    >>> print(target)
    """
    if data_name == "x5_sin":
        filename = os.path.join(root_dir, "data_calculated/x5_sin.csv")
        df = pd.read_csv(filename)
        descriptor_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        target_name = "y" 
        return df, descriptor_names, target_name
    elif data_name == "x5_sin_new":
        filename = os.path.join(root_dir, "data_calculated/x5_sin_new.csv")
        df = pd.read_csv(filename)
        descriptor_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        target_name = "y"         
        return df, descriptor_names, target_name
    elif data_name == "ReCo":
        filename = os.path.join(root_dir, "data/TC_ReCo_detail_descriptor.csv")
        df = pd.read_csv(filename)
        descriptor_names = ['C_R', 'C_T', 'vol_per_atom', 'Z', 'f4', 'd5', 'L4f', 
                            'S4f', 'J4f','(g-1)J4f', '(2-g)J4f']
        target_name = 'Tc'
        return df, descriptor_names, target_name
    elif data_name == "Carbon8":
        filename = os.path.join(root_dir, "data_calculated/Carbon8_cell_descriptor_Etot.csv")
        df = pd.read_csv(filename)
        descriptor_names = ['a0.25_rp1.0', 'a0.25_rp1.5', 'a0.25_rp2.0', 'a0.25_rp2.5',
               'a0.25_rp3.0', 'a0.5_rp1.0', 'a0.5_rp1.5', 'a0.5_rp2.0', 'a0.5_rp2.5',
               'a0.5_rp3.0', 'a1.0_rp1.0', 'a1.0_rp1.5', 'a1.0_rp2.0', 'a1.0_rp2.5',
               'a1.0_rp3.0']
        target_name = "Etot"         
        return df, descriptor_names, target_name
    elif data_name == "Carbon8_minusE":
        df = pd.read_csv(
            "../data_calculated/Carbon8_descriptor_energy.csv", index_col=[0])
        # polytypeを含むデータ
        descriptor_names = ['a0.25_rp1.5', 'a0.25_rp2.5', 'a0.5_rp1.5', 'a0.5_rp2.5',
                            'a1.0_rp1.5', 'a1.0_rp2.5']
        # descriptor_names = descriptor_names[:3]  # 説明変数を制限する．
        target_name = 'minus_energy'
        return df, descriptor_names, target_name
    elif data_name == "ZB_WZ_all":
        filename = os.path.join(root_dir, "data/ZB_WZ_dE_rawdescriptor.csv")
        df = pd.read_csv(filename)
        descriptor_names = ['IP_A', 'EA_A', 'EN_A', 'Highest_occ_A',
                             'Lowest_unocc_A', 'rs_A', 'rp_A', 'rd_A', 'IP_B', 'EA_B', 'EN_B',
                             'Highest_occ_B', 'Lowest_unocc_B', 'rs_B', 'rp_B', 'rd_B']
        target_name = "dE"
        return df, descriptor_names, target_name
    elif data_name == "ZB_WZ_3":
        filename = os.path.join(root_dir, "data_calculated/ZB_WZ_dE_3var.csv")
        df = pd.read_csv(filename)
        descriptor_names = ["desc1" ,"desc2" ,"desc3"]
        target_name = "dE"
        return df, descriptor_names, target_name
    elif data_name == "ZB_WZ_2":
        filename = os.path.join(root_dir, "data_calculated/ZB_WZ_dE_3var.csv")
        df = pd.read_csv(filename)
        descriptor_names = ["desc1" ,"desc2" ]       
        target_name = "dE"
        return df, descriptor_names, target_name
    elif data_name == "Carbon8_desc":
        dirname = f"{root_dir}/data_calculated"
        filename = os.path.join(dirname,
                                "Carbon8_descriptor_selected_sp.csv")
        df_obs = pd.read_csv(filename, index_col=[0, 1])
        descriptor_names = ['a0.25_rp1.0', 'a0.25_rp1.5', 'a0.25_rp2.0',
                            'a0.25_rp2.5', 'a0.25_rp3.0', 'a0.5_rp1.0',
                            'a0.5_rp1.5', 'a0.5_rp2.0',  'a0.5_rp2.5',
                            'a0.5_rp3.0',  'a1.0_rp1.0', 'a1.0_rp1.5',
                            'a1.0_rp2.0', 'a1.0_rp2.5', 'a1.0_rp3.0']
        splabel = "sp_label"
        return df_obs, descriptor_names, splabel
    elif data_name == "Carbon8_desc_all":
        dirname = f"{root_dir}/data_calculated"        
        filename = os.path.join(dirname,
                                "Carbon8_descriptor.csv")
        df_all = pd.read_csv(filename,
                             index_col=[0, 1])
        descriptor_names = ['a0.25_rp1.0', 'a0.25_rp1.5', 'a0.25_rp2.0',
                            'a0.25_rp2.5', 'a0.25_rp3.0', 'a0.5_rp1.0',
                            'a0.5_rp1.5', 'a0.5_rp2.0',  'a0.5_rp2.5',
                            'a0.5_rp3.0',  'a1.0_rp1.0', 'a1.0_rp1.5',
                            'a1.0_rp2.0', 'a1.0_rp2.5', 'a1.0_rp3.0']
        splabel = "sp_label"
        return df_all, descriptor_names, splabel
    elif data_name == "Fe2":
        df = pd.read_csv(f"{root_dir}/data_calculated/Fe2_descriptor.csv")
        splabel = ['a0.70_rp2.40', 'a0.70_rp3.00', 'a0.70_rp3.60',
                            'a0.70_rp4.20', 'a0.70_rp4.80', 'a0.70_rp5.40']
        key_name = "polytype"
        return df, splabel, key_name
    elif data_name == "Fe2_new":
        df = pd.read_csv(f"{root_dir}/data_calculated/Fe2_descriptor_newdata.csv")
        splabel = ['a0.70_rp2.40', 'a0.70_rp3.00', 'a0.70_rp3.60',
                            'a0.70_rp4.20', 'a0.70_rp4.80', 'a0.70_rp5.40']
        key_name = "polytype"
        return df, splabel, key_name
    elif data_name == "mono":
        df = pd.read_csv("../data/mono_structure.csv")
        descriptor_names = ['min_oxidation_state', 'max_oxidation_state', 'row',
                             'group', 's', 'p', 'd', 'f', 'atomic_radius_calculated', 'X', 'IP',
                             'EA']
        target_name = 'crystal_structure'       
        return df, descriptor_names, target_name
    else:
        msg = [f"no {data_name} in the data directory. It may not be an error."]
        raise RuntimeError("\n".join(msg))
