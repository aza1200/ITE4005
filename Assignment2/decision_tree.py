import sys
import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, df, feature_dict, target_name, feature_name=None):
        self.df = df
        self.feature_dict = feature_dict 
        self.target_name = target_name
        self.branches = {}
        self.is_leaf_node = False
        self.prediction = self.df[self.target_name].mode()[0] 
        self.feature_name = feature_name

    def entropy(self, subset):
        probabilities = subset[self.target_name].value_counts(normalize=True)
        entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
        return entropy

    def split_info(self, feature):
        values = self.df[feature].value_counts(normalize=True)
        return -np.sum(values * np.log2(values + np.finfo(float).eps))

    def information_gain(self, feature):
        total_entropy = self.entropy(self.df)
        weights = self.df[feature].value_counts(normalize=True)
        subset_entropy = sum(weights.get(value,0) * self.entropy(self.df[self.df[feature] == value]) for value in self.feature_dict[feature])
        gain = total_entropy - subset_entropy
        return gain / self.split_info(feature)

    def choose_best_feature(self):
        best_feature = None
        max_gain = -1
        for feature in self.feature_dict.keys():
            gain = self.information_gain(feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        
        self.feature_name = best_feature
        return best_feature
    
    def split(self):
        best_feature = self.choose_best_feature()

        if best_feature is None or self.df[self.target_name].nunique()==1 or len(self.feature_dict)==0:
            self.is_leaf_node = True
            self.prediction = self.df[self.target_name].mode()[0]
            return 
        branch_feature_dict = {k: v for k, v in self.feature_dict.items() if k != best_feature}
        
        
        self.branches = {}
        for value in self.feature_dict[best_feature]:
            branch_df = self.df[self.df[best_feature] == value]
            if len(branch_df) > 0: 
                next_tree_node = TreeNode(branch_df, branch_feature_dict, self.target_name)
                self.branches[value] = next_tree_node
            
        for next_tree in self.branches.values():
            next_tree.split()

    def print_tree(self, level=0, prefix="Root: "):
        # Creating indentation and a visual "branch" using "|-- " and "|   " to represent tree branches
        indent = "    " * level
        node_label = f"{self.feature_name}?" if not self.is_leaf_node else f"Predict: {self.prediction}"
        print(f"{indent}{prefix}{node_label}")

        if not self.is_leaf_node:
            for value, branch in self.branches.items():
                branch_prefix = f"{value} -> "
                branch.print_tree(level + 1, branch_prefix)

class DecisionTree:
    def __init__(self):
        self.root = None 

    def fit(self, df):
        self.feature_dict, self.target_dict = self._extract_variables(df)
        self.target_name = df.columns[-1]
        self.root = TreeNode(df, self.feature_dict, self.target_name, "root")
        self.root.split()

    def _extract_variables(self, df):
        feature_dict = {column: df[column].unique().tolist() for column in df.columns[:-1]}
        target_dict = {df.columns[-1]: df[df.columns[-1]].unique().tolist()}
        return feature_dict, target_dict

    def predict(self, df):
        results = []
        for _, row in df.iterrows():
            node = self.root
            # branches = {value1 : node, value2 : node, value3: node}
            while not node.is_leaf_node:
                if row[node.feature_name] in node.branches:
                    node = node.branches[row[node.feature_name]]
                else: break
            results.append(node.prediction)
        df[self.target_name] = results
        return df 

def main(train_fname, test_fname, result_fname):
    train_df = pd.read_csv(train_fname,sep='\t', dtype=str) 
    test_df = pd.read_csv(test_fname,sep='\t', dtype=str)
    
    tree = DecisionTree()
    tree.fit(train_df)

    # tree.root.print_tree()
    result_df = tree.predict(test_df)
    result_df.to_csv(result_fname, sep='\t', index=False)


if __name__ == '__main__':
    train_file_name = str(sys.argv[1])
    test_file_name = str(sys.argv[2])
    result_file_name = str(sys.argv[3])
    main(train_file_name, test_file_name, result_file_name)
