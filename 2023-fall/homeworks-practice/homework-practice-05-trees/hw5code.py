import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def find_best_split(feature_vector : np.ndarray, target_vector : np.ndarray):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    features_sorted = np.sort(feature_vector)
    unique_features = np.unique(features_sorted)
    
    targets_sorted = np.array(target_vector[feature_vector.argsort()])

    thresholds = (unique_features + np.roll(unique_features, -1))[:-1].astype(float) / 2
    
    left_lengths = np.searchsorted(features_sorted, unique_features[:-1], side='right')
    right_lengths = len(features_sorted) - left_lengths
    
    left_cumsum = np.cumsum(targets_sorted)[left_lengths - 1]
    left_p_1 = left_cumsum / left_lengths
    left_p_0 = 1 - left_p_1
    
    right_p_1 = (np.sum(targets_sorted) - left_cumsum) / right_lengths
    right_p_0 = 1 - right_p_1
    
    H_R_l = 1 - left_p_1**2 - left_p_0**2
    H_R_r = 1 - right_p_1**2 - right_p_0**2
    
    ginis = np.array(-(H_R_l * left_lengths + H_R_r * right_lengths) / len(feature_vector))
    # ginis.index = range(len(ginis))
    
    threshold_best = thresholds[ginis.argmax()]
    gini_best = ginis[ginis.argmax()]
    
    return thresholds, ginis, threshold_best, gini_best
    
    


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = None
        self.classes_ =None

    def _fit_node(self, sub_X :np.ndarray, sub_y:np.ndarray, node):
        # починил сменив на равно
        sub_X_arr = np.array(sub_X)
        sub_y_arr = np.array(sub_y)
        
        if self.min_samples_split is not None and sub_X_arr.shape[0] < self.min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y_arr).most_common(1)[0]
        
        if np.all(sub_y_arr == sub_y_arr[0]):
            node["type"] = "terminal"
            node["class"] = sub_y_arr[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        categories_map_best = None
        for feature in range(0, sub_X_arr.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}
            if feature_type == "real":
                feature_vector = sub_X_arr[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X_arr[:, feature])
                clicks = Counter(sub_X_arr[(sub_y_arr == 1).ravel(),feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # починил формулу
                    ratio[key] = current_click / current_count
                    
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                # label encoding by ratio of y=1, everything ok
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X_arr[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1: #no split on same element arr
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y_arr)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = set(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                    categories_map_best = categories_map
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y_arr).most_common(1)[0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            node["categories_map"] = categories_map_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X_arr[split], sub_y_arr[split], node["left_child"])
        # здесь исправил маску
        self._fit_node(sub_X_arr[~split], sub_y_arr[~split], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        cur_node = node
        while cur_node["type"] == "nonterminal":
            feature_best = cur_node["feature_split"]
            feature = x[feature_best]
            if self.feature_types[feature_best] == "real":
                if feature < cur_node["threshold"]:
                    cur_node = cur_node["left_child"]
                else:
                    cur_node = cur_node["right_child"]
            elif self.feature_types[feature_best] == "categorical":
                if feature not in cur_node["categories_map"].keys():
                    left_p = len(cur_node["categories_split"]) / len(cur_node["categories_map"])
                    if np.random.random_sample() < left_p:
                        cur_node = cur_node["left_child"]
                    else:
                        cur_node = cur_node["right_child"]
                else:
                    if feature in cur_node["categories_split"]:
                        cur_node = cur_node["left_child"]
                    else:
                        cur_node = cur_node["right_child"]
            else: raise ValueError()                
        return cur_node["class"]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        X_arr = np.array(X)
        for x in X_arr:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    