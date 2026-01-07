"""
Portfolio optimization engine implementing HRP, CVaR, and MVO models.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.covariance import LedoitWolf
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class PortfolioEngine:
    """
    A portfolio optimization engine implementing HRP, CVaR, and MVO models.
    
    This class provides three different portfolio optimization approaches:
    1. Hierarchical Risk Parity (HRP)
    2. CVaR Optimization
    3. Mean-Variance Optimization (MVO)
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize the PortfolioEngine with return data.
        
        Args:
            returns: DataFrame with asset returns
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Calculate covariance matrix using Ledoit-Wolf shrinkage
        self.cov_matrix = self._ledoit_wolf_covariance(returns.values)
        self.correlation_matrix = self._cov_to_corr(self.cov_matrix)
        
        # Calculate expected returns
        self.expected_returns = returns.mean().values
        
    def _ledoit_wolf_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate the Ledoit-Wolf shrinkage covariance matrix.
        
        Args:
            returns: Array of asset returns
            
        Returns:
            Shrunk covariance matrix
        """
        lw = LedoitWolf()
        lw.fit(returns)
        return lw.covariance_
    
    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Convert covariance matrix to correlation matrix.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Correlation matrix
        """
        stds = np.sqrt(np.diag(cov_matrix))
        return cov_matrix / np.outer(stds, stds)
    
    def _get_quasi_diagonal(self, link_matrix: np.ndarray) -> list:
        """
        Calculate the quasi-diagonalization of the linkage matrix.
        
        Args:
            link_matrix: Linkage matrix from hierarchical clustering
            
        Returns:
            Quasi-diagonalized asset indices
        """
        # Start with leaf nodes and build the quasi-diagonalized order
        n = link_matrix.shape[0] + 1
        order = [0]  # Start with first asset
        
        for i in range(link_matrix.shape[0]):
            # Find the cluster that was formed at step i
            cluster = int(link_matrix[i, 0])
            if cluster < n:
                # It's an original asset
                order.append(cluster)
            else:
                # It's a previously formed cluster
                idx = cluster - n
                # Insert the new cluster in the right position
                order = [int(link_matrix[idx, 0]) if x == cluster else x for x in order]
                order = [int(link_matrix[idx, 1]) if x == cluster else x for x in order]
        
        # More direct approach for quasi-diagonalization
        order = self._compute_quasi_diagonal(link_matrix)
        return order
    
    def _compute_quasi_diagonal(self, link_matrix: np.ndarray) -> list:
        """
        Compute the quasi-diagonalized order of assets using the linkage matrix.
        
        Args:
            link_matrix: Linkage matrix from hierarchical clustering
            
        Returns:
            Ordered list of asset indices
        """
        n = link_matrix.shape[0] + 1
        order = list(range(n))
        
        for i in range(link_matrix.shape[0]):
            # Merge clusters at step i
            cluster1 = int(link_matrix[i, 0])
            cluster2 = int(link_matrix[i, 1])
            
            # Find positions of these clusters in the current order
            pos1 = None
            pos2 = None
            for j, val in enumerate(order):
                if val == cluster1:
                    pos1 = j
                elif val == cluster2:
                    pos2 = j
            
            # Replace the two clusters with the new cluster (represented by its index)
            new_cluster_idx = n + i
            if pos1 is not None and pos2 is not None:
                # Replace both with the new cluster index
                order = [new_cluster_idx if x in [cluster1, cluster2] else x for x in order]
            elif pos1 is not None:
                order[pos1] = new_cluster_idx
                if pos2 is not None:
                    order[pos2] = new_cluster_idx
        
        # Convert back to original asset indices
        return [x for x in range(n)]
    
    def _get_cluster_var(self, cov_matrix: np.ndarray, cluster_indices: list) -> float:
        """
        Calculate the variance of a cluster of assets.
        
        Args:
            cov_matrix: Covariance matrix
            cluster_indices: List of asset indices in the cluster
            
        Returns:
            Variance of the cluster
        """
        cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
        weights = np.ones(len(cluster_indices)) / len(cluster_indices)
        return np.dot(weights, np.dot(cluster_cov, weights))
    
    def _recursive_bisection(self, cov_matrix: np.ndarray, ordered_indices: list) -> np.ndarray:
        """
        Perform recursive bisection to calculate HRP weights.
        
        Args:
            cov_matrix: Covariance matrix
            ordered_indices: Ordered list of asset indices
            
        Returns:
            Array of portfolio weights
        """
        weights = np.zeros(len(ordered_indices))
        clusters = [ordered_indices]
        
        while clusters:
            cluster = clusters.pop(0)
            if len(cluster) == 1:
                # Single asset - assign its weight
                weights[cluster[0]] = 1.0
            else:
                # Split cluster in half
                mid = len(cluster) // 2
                cluster1 = cluster[:mid]
                cluster2 = cluster[mid:]
                
                # Calculate variance for each sub-cluster
                var1 = self._get_cluster_var(cov_matrix, cluster1)
                var2 = self._get_cluster_var(cov_matrix, cluster2)
                
                # Calculate allocation weights for sub-clusters
                inv_var1 = 1.0 / var1 if var1 != 0 else 0
                inv_var2 = 1.0 / var2 if var2 != 0 else 0
                
                if inv_var1 + inv_var2 != 0:
                    alloc1 = inv_var1 / (inv_var1 + inv_var2)
                    alloc2 = inv_var2 / (inv_var1 + inv_var2)
                else:
                    alloc1 = alloc2 = 0.5
                
                # Assign weights to sub-clusters
                for idx in cluster1:
                    weights[idx] *= alloc1
                for idx in cluster2:
                    weights[idx] *= alloc2
                
                # Add sub-clusters to processing queue
                if len(cluster1) > 1:
                    clusters.append(cluster1)
                else:
                    # Single asset - assign its portion of the allocation
                    weights[cluster1[0]] = alloc1 if len(cluster) == len(ordered_indices) else weights[cluster1[0]] * alloc1
                
                if len(cluster2) > 1:
                    clusters.append(cluster2)
                else:
                    # Single asset - assign its portion of the allocation
                    weights[cluster2[0]] = alloc2 if len(cluster) == len(ordered_indices) else weights[cluster2[0]] * alloc2
        
        # A more direct implementation of HRP recursive bisection
        weights = np.ones(len(ordered_indices))
        clusters = [(0, len(ordered_indices))]
        
        while clusters:
            start, end = clusters.pop(0)
            if end - start <= 1:
                continue  # Single asset, no more bisection needed
            
            # Split the cluster in half
            mid = (start + end) // 2
            
            # Calculate weights for each half based on inverse variance
            left_indices = ordered_indices[start:mid]
            right_indices = ordered_indices[mid:end]
            
            # Calculate the variance of each cluster
            left_cov = cov_matrix[np.ix_(left_indices, left_indices)]
            right_cov = cov_matrix[np.ix_(right_indices, right_indices)]
            
            # Use inverse variance allocation
            left_var = np.sum(left_cov) / len(left_indices)**2  # Simplified variance proxy
            right_var = np.sum(right_cov) / len(right_indices)**2
            
            # Calculate allocation weights
            if left_var + right_var > 0:
                left_weight = right_var / (left_var + right_var)
                right_weight = left_var / (left_var + right_var)
            else:
                left_weight = right_weight = 0.5
            
            # Apply weights to the assets in each cluster
            for i in range(start, mid):
                weights[ordered_indices[i]] *= left_weight
            for i in range(mid, end):
                weights[ordered_indices[i]] *= right_weight
            
            # Add sub-clusters for further processing
            if mid - start > 1:
                clusters.append((start, mid))
            if end - mid > 1:
                clusters.append((mid, end))
        
        return weights
    
    def hrp_optimization(self) -> Dict[str, float]:
        """
        Perform Hierarchical Risk Parity (HRP) optimization.
        
        This method implements the 3 stages of HRP:
        1. Tree clustering (using correlation matrix)
        2. Quasi-diagonalization
        3. Recursive bisection allocation
        
        Returns:
            Dictionary with asset weights
        """
        # Step 1: Tree clustering using correlation matrix
        # Convert correlation to distance matrix
        distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))
        
        # Perform hierarchical clustering
        # Use numeric indices instead of string indexing
        link_matrix = linkage(distance_matrix, method='single')
        
        # Step 2: Quasi-diagonalization
        # Get the order of assets after quasi-diagonalization
        ordered_indices = list(range(self.n_assets))  # Simple implementation
        
        # Actually perform clustering and ordering
        sorted_corr = pd.DataFrame(distance_matrix, 
                                   index=self.assets, 
                                   columns=self.assets)
        
        # Use scipy's dendrogram to get the correct order
        dend = dendrogram(link_matrix, labels=self.assets, no_plot=True)
        ordered_indices = dend['leaves']
        
        # Step 3: Recursive bisection allocation
        weights = self._recursive_bisection(self.cov_matrix, ordered_indices)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return dict(zip(self.assets, weights))
    
    def cvar_optimization(self, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform CVaR (Conditional Value at Risk) optimization.
        
        This method minimizes the Expected Shortfall at the given confidence level.
        
        Args:
            alpha: Confidence level (e.g., 0.05 for 95% confidence)
            
        Returns:
            Dictionary with asset weights
        """
        n = self.n_assets
        
        # Objective function: minimize CVaR
        def objective(weights):
            portfolio_returns = self.returns.values @ weights
            # Calculate VaR (Value at Risk) at alpha level
            var = np.percentile(portfolio_returns, alpha * 100)
            # Calculate CVaR (expected loss beyond VaR)
            cvar = -np.mean(portfolio_returns[portfolio_returns <= var])
            return cvar
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
        ]
        
        # Bounds for weights (0 to 1 for long-only)
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess: equal weights
        init_guess = np.array([1 / n] * n)
        
        # Optimize
        result = minimize(objective, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            print(f"CVaR optimization failed: {result.message}")
            # Fallback to equal weights
            weights = np.array([1 / n] * n)
        else:
            weights = result.x
        
        return dict(zip(self.assets, weights))
    
    def mvo_optimization(self, target_return: float = None) -> Dict[str, float]:
        """
        Perform Mean-Variance Optimization (Modern Portfolio Theory).
        
        This method implements the standard MVO to maximize Sharpe ratio.
        
        Args:
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            
        Returns:
            Dictionary with asset weights
        """
        n = self.n_assets
        
        if target_return is None:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(weights * self.expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
                portfolio_vol = np.sqrt(portfolio_variance)
                # Negative Sharpe ratio to minimize (assuming risk-free rate = 0)
                return -portfolio_return / portfolio_vol if portfolio_vol > 0 else np.inf
        
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
            ]
        else:
            # Minimize variance for a given return
            def objective(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
                {'type': 'eq', 'fun': lambda w: np.sum(w * self.expected_returns) - target_return},  # Target return
            ]
        
        # Bounds for weights (0 to 1 for long-only)
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess: equal weights
        init_guess = np.array([1 / n] * n)
        
        # Optimize
        result = minimize(objective, init_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            print(f"MVO optimization failed: {result.message}")
            # Fallback to equal weights
            weights = np.array([1 / n] * n)
        else:
            weights = result.x
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        return dict(zip(self.assets, weights))
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float], 
                                  returns: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate portfolio metrics based on given weights.
        
        Args:
            weights: Dictionary with asset weights
            returns: DataFrame with returns (defaults to self.returns)
            
        Returns:
            Dictionary with portfolio metrics
        """
        if returns is None:
            returns = self.returns
            
        # Convert weights to array in the same order as returns columns
        weight_array = np.array([weights[asset] for asset in self.assets])
        
        # Calculate portfolio returns
        portfolio_returns = returns @ weight_array
        
        # Calculate metrics
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        sharpe_ratio = mean_return / volatility if volatility != 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cumulative_return': cumulative_returns.iloc[-1] - 1
        }