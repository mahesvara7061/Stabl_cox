import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


def remove_low_info_samples(X, threshold=1.0):
    """Removes low info samples

    A sample is considered to have sufficient info if the nan fraction is below the
    input hard_threshold.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_repeats, n_features)
        Data from which to compute NaN proportion, where `n_repeats` is
        the number of samples and `n_features` is the number of features.

    threshold : float, default=1.0
        Samples with a proportion of NaN greater than this value will be removed.
    
    Returns
    -------
    X_reduced : array, shape(n_samples_out, n_features)
        The reduced array of siwe n_samples_out, n_features
    """
    if not isinstance(threshold, float) or (threshold < 0. or threshold > 1.):
        raise ValueError(f"Nan fraction must be between 0 and 1 Got: {threshold}")

    nan_fraction = np.isnan(X).sum(1) / X.shape[1]
    mask = nan_fraction < threshold
    return X[mask]


# class LowInfoFilter(SelectorMixin, BaseEstimator):
#     """Feature selector that removes all low-variance features.

#     This feature selection algorithm looks only at the features (X), not the
#     desired outputs (y), and can thus be used for unsupervised learning.

#     A feature is considered to be a low info one if the proportion of nan
#     values is above a given hard_threshold set by the user.

#     Parameters
#     ----------
#     max_nan_fraction : float, default=0.2
#         Features with a proportion of nan values greater than this hard_threshold will
#         be removed. By default, the proportion is set to 0.2.

#     Attributes
#     ----------
#     nan_counts_ : array, shape (n_features,)
#         Count of nan values for each individual feature.

#     n_features_in_ : int
#         Number of features seen during fit.

#     feature_names_in_ : ndarray of shape (n_features_in_, )
#         Names of features seen during the fit. Defined only when X
#         has feature names that are all strings.

#     Notes
#     -----
#     Allows NaN in the input.
#     Raises ValueError if no feature in X meets the low info hard_threshold.
#     """

#     def __init__(self, max_nan_fraction=0.2):
#         self.max_nan_fraction = max_nan_fraction
#         self.n_samples = None
#         self.nan_counts_ = None

#     def fit(self, X, y=None):
#         """Learn empirical Nan proportion in X.

#         Parameters
#         ----------
#         X : {array-like, sparse matrix}, shape (n_repeats, n_features)
#             Data from which to compute NaN proportion, where `n_repeats` is
#             the number of samples and `n_features` is the number of features.

#         y : any, default=None
#             Ignored. This parameter exists only for compatibility with
#             sklearn.pipeline.Pipeline.

#         Returns
#         -------
#         self : object
#             Returns the instance itself.
#         """
#         X = self._validate_data(
#             X,
#             accept_sparse=("csr", "csc"),
#             dtype=np.float64,
#             force_all_finite="allow-nan",
#         )

#         if self.max_nan_fraction > 1 or self.max_nan_fraction < 0:
#             raise ValueError(
#                 f"Nan fraction must be between 0 and 1 Got: {self.max_nan_fraction}")

#         n_samples = X.shape[0]
#         self.n_samples = n_samples
#         self.nan_counts_ = np.isnan(np.array(X)).sum(0)

#         if np.all(~np.isfinite(self.nan_counts_) | (
#                 self.nan_counts_ > self.max_nan_fraction * self.n_samples)):
#             msg = "No feature in X meets the low info hard_threshold {0:.5f}"
#             if n_samples == 1:
#                 msg += " (X contains only one sample)"
#             raise ValueError(msg.format(self.max_nan_fraction))

#         return self

#     def _get_support_mask(self):
#         """Get a mask, or integer index, of the features selected
            
#         Returns
#         -------
#         support : array
#             An index that selects the retained features from a feature vector.
#             This is a boolean array of shape
#             [# input features], in which an element is True iff its
#             corresponding feature is selected for retention. 
#         """
#         check_is_fitted(self)

#         return self.nan_counts_ <= self.max_nan_fraction * self.n_samples

#     def _more_tags(self):
#         # Useful to allow the use of nan values
#         # For the transform function ;)
#         return {"allow_nan": True}
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

class LowInfoFilter(SelectorMixin, BaseEstimator):
    """
    Loại bỏ các feature 'ít thông tin' dựa trên tỷ lệ NaN.
    Một feature bị loại nếu tỷ lệ NaN > max_nan_fraction.

    Parameters
    ----------
    max_nan_fraction : float, default=0.2
        Ngưỡng tối đa cho tỷ lệ NaN trên mỗi cột.

    Attributes
    ----------
    nan_counts_ : array of shape (n_features,)
        Số lượng NaN cho từng cột.
    n_features_in_ : int
        Số cột đầu vào lúc fit.
    feature_names_in_ : ndarray[str] hoặc None
        Tên cột nếu đầu vào là DataFrame, dùng để xuất tên cột sau lọc.
    """

    def __init__(self, max_nan_fraction=0.2):
        self.max_nan_fraction = float(max_nan_fraction)
        self.n_samples = None
        self.nan_counts_ = None
        self.feature_names_in_ = None
        self._n_features = None  # nội bộ

    def fit(self, X, y=None):
        # Chấp nhận cả DataFrame và ndarray; không dùng _validate_data để tránh lỗi phiên bản
        if isinstance(X, pd.DataFrame):
            Xv = X.to_numpy(dtype=float, copy=False)
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        else:
            # ép về ndarray float; cho phép NaN
            Xv = np.asarray(X, dtype=float)
            self.feature_names_in_ = None

        if not (0.0 <= self.max_nan_fraction <= 1.0):
            raise ValueError(
                f"Nan fraction must be between 0 and 1. Got: {self.max_nan_fraction}"
            )

        if Xv.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {Xv.shape}")

        n_samples, n_features = Xv.shape
        self.n_samples = int(n_samples)
        self._n_features = int(n_features)

        # Đếm NaN theo cột
        self.nan_counts_ = np.isnan(Xv).sum(axis=0)

        # Nếu tất cả cột đều vượt ngưỡng → báo lỗi (giống hành vi cũ)
        if np.all(self.nan_counts_ > self.max_nan_fraction * self.n_samples):
            msg = "No feature in X meets the low info threshold ({0:.5f})."
            if n_samples == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.max_nan_fraction))

        return self

    def _get_support_mask(self):
        check_is_fitted(self, attributes=["nan_counts_", "n_samples"])
        # Giữ lại cột có tỷ lệ NaN <= ngưỡng
        support = self.nan_counts_ <= self.max_nan_fraction * self.n_samples

        # Phòng trường hợp hiếm: tất cả False → giữ lại vài cột ít NaN nhất
        if not np.any(support) and self._n_features and self._n_features > 0:
            k = max(1, int(0.01 * self._n_features))
            keep_idx = np.argsort(self.nan_counts_)[:k]
            support = np.zeros(self._n_features, dtype=bool)
            support[keep_idx] = True

        return support

    def get_feature_names_out(self, input_features=None):
        """
        Trả về tên cột sau khi lọc (để Pipeline.get_feature_names_out() hoạt động).
        """
        mask = self.get_support()
        if input_features is None:
            if self.feature_names_in_ is not None:
                in_feats = self.feature_names_in_
            else:
                in_feats = np.array([f"x{i}" for i in range(mask.shape[0])], dtype=object)
        else:
            in_feats = np.asarray(input_features, dtype=object)
        return in_feats[mask]

    def _more_tags(self):
        # Cho phép NaN
        return {"allow_nan": True}
