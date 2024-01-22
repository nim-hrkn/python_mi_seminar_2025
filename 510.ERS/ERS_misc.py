

import itertools
import numpy as np
from mass_function import MassFunction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_evidence(alloy_i, alloy_j):
    """
    alloy_i = ["Ag", "Fe"] ，
    alloy_j = ["Cu", "Fe"]として
    extract_evidence(alloy_i, alloy_j)は
    共通分があれば非共通要素 (frozenset({'Ag'}), frozenset({'Cu'}))を返す．

    ['Fe', 'Ag']と ['Ru', 'Pd'] では None　を返す．

    Args:
        alloy_i (List[str]): 物質1.
        alloy_j (List[str]): 物質2.

    Returns:
        tuple containg

        frozenset: alloy_i 非共通要素.
        frozenset: alloy_j 非共通要素.
    """
    intersection_ai_aj = alloy_i.intersection(alloy_j)  # Intersection of A_i and A_j
    if len(intersection_ai_aj) == 0:  # Extract evidence from Ai and Aj that their intersection is not empty
        return None
    else:
        combination_t = alloy_i - alloy_j  # C_t = A_i - A_j
        combination_v = alloy_j - alloy_i  # C_v = A_j - A_i
        if len(combination_t) <= 2 and len(combination_v) <= 2:  # Extract evidence from Ai and Aj that number of substituted element does not exceed 2 elements
            return combination_t, combination_v
        else:
            return None


def extract_HEA_evidence(alloy_new, alloy_k):
    if len(alloy_k.intersection(alloy_new)) != 0:
        combination_t = alloy_k - alloy_new
        combination_v = alloy_new - alloy_k
        if len(combination_t) <= 2 and len(combination_v) <= 2:
            return combination_t, combination_v
        else:
            return None
    return None


def make_mass_function(alloy_i, alloy_j, samelabel, evidence, alpha, debug=False):
    """make mass function of evicence.

    Args:
        alloy_i (frozenset): a list of element of alloy_i.
        alloy_j (frozenset): a list of element of alloy_j.
        evidence (Tuple[frozenset]): alloy_iとalloy_jの非共通元素 
        samelabel (bool): both are the same character or not.
        alpha (float): alpha value of mass function.
        debug (bool, optional): debug print or not. Defaults to False.

    Returns:_description_nt j
    """
    if evidence is not None:
        combination_t, combination_v = evidence
        if samelabel:
            mass_function_1 = MassFunction(source=[({"similar"}, alpha)],
                                           coreset={"similar", "dissimilar"})
        else:
            mass_function_1 = MassFunction(source=[({"dissimilar"}, alpha)],
                                           coreset={"similar", "dissimilar"})

        return mass_function_1
    else:
        None


def make_similarity_df(elements, mass_functions):
    """make similarity dataframes


    Args:
        elements (List[str]): a list of elmemnts.
        mass_functions (List[MassFunction]): a list of MassFunction.

    Returns:
        Tuple containing

        pd.DataFrame: m({similarity}).
        pd.DataFrame: m({dissimilarity}).
        pd.DataFrame: m({similarity, dissimilarity}).
    """
    subsets = []
    for size_subset in range(3):
        for subset in itertools.combinations(elements, size_subset):
            subsets.append(subset)
    n_subsets = len(subsets)
    columns_name = ["|".join(sorted(k)) for k in subsets]
    print(columns_name)
    similarity_matrix = np.zeros((n_subsets, n_subsets))
    np.fill_diagonal(similarity_matrix, 1)
    df_similarity = pd.DataFrame(similarity_matrix, columns=columns_name, index=columns_name)

    dissimilarity_matrix = np.zeros((n_subsets, n_subsets))
    df_dissimilarity = pd.DataFrame(dissimilarity_matrix, columns=columns_name, index=columns_name)

    unknown_matrix = np.ones((n_subsets, n_subsets))
    np.fill_diagonal(unknown_matrix, 0)
    df_unknown = pd.DataFrame(unknown_matrix, columns=columns_name, index=columns_name)

    for combination_t, combination_v, mass_function in mass_functions:
        index_t, index_v = "|".join(sorted(combination_t)), "|".join(sorted(combination_v))
        print(index_t, index_v)
        similar_score = df_similarity.loc[index_t, index_v]
        dissimilar_score = df_dissimilarity.loc[index_t, index_v]
        unk_score = df_unknown.loc[index_t, index_v]
        combined_mass_function = mass_function.combine(
            MassFunction(
                source=[({"similar"}, similar_score),
                        ({"dissimilar"}, dissimilar_score),
                        ({"similar", "dissimilar"}, unk_score)], coreset={"similar", "dissimilar"}
            )
        )
        # Update matrices
        df_similarity.loc[index_t, index_v] = combined_mass_function[frozenset({"similar"})]
        df_dissimilarity.loc[index_t, index_v] = combined_mass_function[frozenset({"dissimilar"})]
        df_unknown.loc[index_t, index_v] = combined_mass_function[frozenset({"similar", "dissimilar"})]
        df_similarity.loc[index_v, index_t] = combined_mass_function[frozenset({"similar"})]
        df_dissimilarity.loc[index_v, index_t] = combined_mass_function[frozenset({"dissimilar"})]
        df_unknown.loc[index_v, index_t] = combined_mass_function[frozenset({"similar", "dissimilar"})]
    return df_similarity, df_dissimilarity, df_unknown


def plot_final_decisions(candidates, final_decisions):
    """plot final decisions.

    Args:
        candidates (List[List[str]]): material candiates.
        final_decisions (List[MassFunction]): a list of mass functions.
    """
    materials = []
    positive_votes = []
    negative_votes = []
    unknown_votes = []
    for candidate, final_decision in zip(candidates, final_decisions):
        materials.append("".join(candidate))
        positive_votes.append(final_decision[frozenset({'HEA'})])
        negative_votes.append(final_decision[frozenset({"not_HEA"})])
        unknown_votes.append(final_decision[frozenset({'HEA', "not_HEA"})])

    fig = plt.figure(figsize=(8, 8), dpi=300)
    p1 = plt.barh(materials, positive_votes, .5, color="cornflowerblue", label="Forming HEA")
    p2 = plt.barh(materials, unknown_votes, .5, label="Unknown",
                  left=positive_votes, color="gainsboro")
    p3 = plt.barh(materials, negative_votes, .5, label="Not forming HEA",
                  left=[p + u for p, u in zip(positive_votes, unknown_votes)], color="firebrick")
    for y, (x1, x2, x3) in enumerate(zip(positive_votes, unknown_votes, negative_votes)):
        if x1 > 0.08:
            plt.text(x1/2, y, "{}%".format(round(x1*100, 1)),
                     fontsize=16, ha='center', va='center',
                     weight="bold", color="white"
                     )
        if x2 > 0.08:
            plt.text(x1+x2/2, y, "{}%".format(round(x2*100, 1)),
                     fontsize=16, ha='center', va='center',
                     weight="bold"
                     )
        if x3 > 0.08:
            plt.text(x1+x2+x3/2, y, "{}%".format(round(x3*100, 1)),
                     fontsize=16, ha='center', va='center',
                     weight="bold", color="white"
                     )

    fig.axes[0].get_xaxis().set_visible(False)
    plt.legend(bbox_to_anchor=(-0.015, 1), ncol=3, loc='lower left', fontsize=14, columnspacing=1.6)
    plt.xticks(np.arange(0.0, 1.01, 0.2))
    plt.ylim(ymin=-0.5, ymax=len(materials)-0.5)
    plt.xlim(xmin=0, xmax=1)
    plt.style.use('default')
    plt.tick_params(axis='y', which='major', labelsize=20)


def plot_similarity_matrix(df_similarity, elements):
    """plot similarity matrix.

    Args:
        df_similarity (pd.DataFrame): similarity matrix.
        elements (List[str]): a list of elements.    
    """
    plt.figure(figsize=(10, 8), dpi=300)
    plt.style.use('default')
    plt.tick_params(axis='x', which='major', labelsize=24)
    plt.tick_params(axis='y', which='major', labelsize=24)
    # Sort the order of elements in similarity matrix using the order obtained from dendogram
    # df_matrix = sort_matrix(df_similarity.loc[ELEMENTS, ELEMENTS], order=dendrogram["ivl"])
    df_matrix = df_similarity.loc[elements, elements]
    ax = sns.heatmap(df_matrix, cmap="YlGnBu", xticklabels=df_matrix.columns.values,
                     yticklabels=df_matrix.index.values, vmax=1, vmin=0
                     )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_title("Similarity score")
    plt.xticks(rotation=45, horizontalalignment="right", fontsize=12)
    plt.yticks(horizontalalignment="right", fontsize=12)
