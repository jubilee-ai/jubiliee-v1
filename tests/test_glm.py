"""
Tests for GLM (Generalized Linear Model) Training Tool

Tests Poisson, Gamma, and Tweedie distributions for insurance pricing.
"""

import sys
import os

# Add training module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models-tools", "training"))

import numpy as np
import pandas as pd
import pytest

from glm import GLMTrainingInput, train_glm, sklearn_glm_tool
from model_storage import get_model_info, load_model, delete_model


class TestPoissonRegression:
    """Tests for Poisson GLM (claim frequency modeling)."""

    @pytest.fixture
    def claim_frequency_data(self):
        """Generate synthetic claim frequency data."""
        np.random.seed(42)
        data = []
        for _ in range(100):
            driver_age = int(np.random.uniform(18, 70))
            years_licensed = max(0, driver_age - 18 + int(np.random.normal(0, 2)))
            vehicle_age = int(np.random.uniform(0, 15))
            annual_mileage = int(np.random.uniform(5000, 30000))
            
            # Base rate affected by risk factors
            base_rate = 0.15
            base_rate += 0.008 * max(0, 25 - driver_age)  # Young drivers
            base_rate += 0.005 * max(0, 5 - years_licensed)  # Inexperienced
            base_rate += 0.003 * vehicle_age  # Older cars
            base_rate += 0.000002 * annual_mileage  # More driving
            
            num_claims = int(np.random.poisson(base_rate * 2))
            
            data.append({
                'driver_age': driver_age,
                'years_licensed': years_licensed,
                'vehicle_age': vehicle_age,
                'annual_mileage': annual_mileage,
                'num_claims': num_claims,
            })
        return data

    def test_poisson_training(self, claim_frequency_data):
        """Test Poisson regression training for claim counts."""
        input_data = GLMTrainingInput(
            model_name="test_poisson_freq",
            description="Poisson model for claim frequency",
            data=claim_frequency_data,
            target_column="num_claims",
            distribution="poisson",
            alpha=1.0,
            test_size=0.2,
            random_state=42,
        )
        
        result = train_glm(input_data)
        
        assert result.success is True
        assert result.distribution == "poisson"
        assert result.n_samples == 100
        assert result.n_features == 4
        assert 0 <= result.train_deviance <= 1
        assert result.saved_path.endswith(".joblib")
        assert len(result.coefficients) == 4
        
        # Cleanup
        delete_model("test_poisson_freq")

    def test_poisson_tool_invocation(self, claim_frequency_data):
        """Test Poisson GLM via LangChain tool."""
        result = sklearn_glm_tool.invoke({
            'model_name': 'test_poisson_tool',
            'description': 'Test Poisson via tool',
            'data': claim_frequency_data,
            'target_column': 'num_claims',
            'distribution': 'poisson',
        })
        
        assert "TRAINING COMPLETE" in result
        assert "POISSON" in result
        assert "D²" in result or "deviance" in result.lower()
        
        # Verify model was registered
        info = get_model_info("test_poisson_tool")
        assert info is not None
        assert info["model_type"] == "sklearn_glm"
        
        delete_model("test_poisson_tool")


class TestGammaRegression:
    """Tests for Gamma GLM (claim severity modeling)."""

    @pytest.fixture
    def claim_severity_data(self):
        """Generate synthetic claim severity data (all positive values)."""
        np.random.seed(123)
        data = []
        for _ in range(80):
            vehicle_value = int(np.random.uniform(15000, 80000))
            repair_complexity = np.random.choice(['low', 'medium', 'high'])
            body_shop_region = np.random.choice(['urban', 'suburban', 'rural'])
            
            # Base severity
            base = 2000 + vehicle_value * 0.03
            if repair_complexity == 'medium':
                base *= 1.3
            elif repair_complexity == 'high':
                base *= 1.8
            if body_shop_region == 'urban':
                base *= 1.2
            
            # Gamma-distributed claim amount (always positive)
            claim_amount = max(100, base + np.random.gamma(2, base * 0.1))
            
            data.append({
                'vehicle_value': vehicle_value,
                'repair_complexity': repair_complexity,
                'body_shop_region': body_shop_region,
                'claim_amount': round(claim_amount, 2),
            })
        return data

    def test_gamma_training(self, claim_severity_data):
        """Test Gamma regression for claim severity."""
        input_data = GLMTrainingInput(
            model_name="test_gamma_severity",
            description="Gamma model for claim severity",
            data=claim_severity_data,
            target_column="claim_amount",
            distribution="gamma",
            categorical_columns=['repair_complexity', 'body_shop_region'],
            alpha=0.5,
            random_state=42,
        )
        
        result = train_glm(input_data)
        
        assert result.success is True
        assert result.distribution == "gamma"
        assert result.target_stats['min'] > 0  # Gamma requires positive values
        assert result.target_stats['zeros_pct'] == 0.0
        
        delete_model("test_gamma_severity")

    def test_gamma_predictions(self, claim_severity_data):
        """Test predictions with trained Gamma model."""
        sklearn_glm_tool.invoke({
            'model_name': 'test_gamma_pred',
            'data': claim_severity_data,
            'target_column': 'claim_amount',
            'distribution': 'gamma',
            'categorical_columns': ['repair_complexity', 'body_shop_region'],
        })
        
        # Load and predict
        model = load_model("test_gamma_pred")
        test_df = pd.DataFrame([
            {'vehicle_value': 50000, 'repair_complexity': 'high', 'body_shop_region': 'urban'},
            {'vehicle_value': 20000, 'repair_complexity': 'low', 'body_shop_region': 'rural'},
        ])
        
        predictions = model.predict(test_df)
        assert len(predictions) == 2
        assert all(p > 0 for p in predictions)  # Gamma predictions are positive
        assert predictions[0] > predictions[1]  # High value/complexity > low
        
        delete_model("test_gamma_pred")


class TestTweedieRegression:
    """Tests for Tweedie GLM (pure premium / total cost modeling)."""

    @pytest.fixture
    def pure_premium_data(self):
        """Generate synthetic pure premium data (many zeros, some positive)."""
        np.random.seed(456)
        data = []
        for _ in range(120):
            age = int(np.random.uniform(20, 70))
            coverage_level = np.random.choice(['basic', 'standard', 'premium'])
            deductible = np.random.choice([500, 1000, 2000])
            
            # Probability of claim
            claim_prob = 0.15 + 0.005 * max(0, 30 - age)
            if coverage_level == 'premium':
                claim_prob *= 0.8  # Better drivers choose premium
            
            # Total cost (many zeros for no-claim)
            if np.random.random() < claim_prob:
                base_cost = 3000 - deductible + np.random.gamma(3, 1000)
                total_cost = max(0, base_cost)
            else:
                total_cost = 0
            
            data.append({
                'age': age,
                'coverage_level': coverage_level,
                'deductible': deductible,
                'total_cost': round(total_cost, 2),
            })
        return data

    def test_tweedie_training(self, pure_premium_data):
        """Test Tweedie regression for pure premium."""
        input_data = GLMTrainingInput(
            model_name="test_tweedie_premium",
            description="Tweedie model for pure premium",
            data=pure_premium_data,
            target_column="total_cost",
            distribution="tweedie",
            tweedie_power=1.5,
            categorical_columns=['coverage_level'],
            random_state=42,
        )
        
        result = train_glm(input_data)
        
        assert result.success is True
        assert result.distribution == "tweedie"
        assert result.target_stats['zeros_pct'] > 0  # Tweedie handles zeros
        assert result.target_stats['min'] >= 0
        
        delete_model("test_tweedie_premium")

    def test_tweedie_power_parameter(self, pure_premium_data):
        """Test Tweedie with different power parameters."""
        for power in [1.2, 1.5, 1.8]:
            result = sklearn_glm_tool.invoke({
                'model_name': f'test_tweedie_p{int(power*10)}',
                'data': pure_premium_data,
                'target_column': 'total_cost',
                'distribution': 'tweedie',
                'tweedie_power': power,
            })
            
            assert "TRAINING COMPLETE" in result
            assert f"power={power}" in result
            
            delete_model(f'test_tweedie_p{int(power*10)}')


class TestGLMValidation:
    """Tests for input validation and error handling."""

    def test_poisson_negative_target_fails(self):
        """Poisson should fail with negative target values."""
        bad_data = [
            {'x': 1, 'count': -1},
            {'x': 2, 'count': 3},
        ]
        
        result = sklearn_glm_tool.invoke({
            'model_name': 'test_bad_poisson',
            'data': bad_data,
            'target_column': 'count',
            'distribution': 'poisson',
        })
        
        assert "FAILED" in result or "Error" in result

    def test_gamma_zero_target_fails(self):
        """Gamma should fail with zero or negative target values."""
        bad_data = [
            {'x': 1, 'amount': 0},
            {'x': 2, 'amount': 100},
        ]
        
        result = sklearn_glm_tool.invoke({
            'model_name': 'test_bad_gamma',
            'data': bad_data,
            'target_column': 'amount',
            'distribution': 'gamma',
        })
        
        assert "FAILED" in result or "Error" in result

    def test_missing_target_column(self):
        """Should fail if target column doesn't exist."""
        data = [{'x': 1, 'y': 2}]
        
        result = sklearn_glm_tool.invoke({
            'model_name': 'test_missing_col',
            'data': data,
            'target_column': 'nonexistent',
            'distribution': 'poisson',
        })
        
        assert "FAILED" in result or "not found" in result


class TestGLMIntegration:
    """Integration tests for full GLM workflow."""

    def test_full_pricing_workflow(self):
        """Test complete insurance pricing workflow with GLM."""
        np.random.seed(789)
        
        # Generate realistic auto insurance data
        data = []
        for _ in range(150):
            age = int(np.random.uniform(18, 75))
            gender = np.random.choice(['M', 'F'])
            territory = np.random.choice(['urban', 'suburban', 'rural'])
            vehicle_age = int(np.random.uniform(0, 15))
            
            # Frequency (Poisson)
            freq_base = 0.1
            freq_base *= (1.5 if age < 25 else 0.8 if age > 60 else 1.0)
            freq_base *= (1.2 if territory == 'urban' else 0.9 if territory == 'rural' else 1.0)
            num_claims = int(np.random.poisson(freq_base))
            
            # Severity (only if claims > 0)
            if num_claims > 0:
                sev_base = 3000 + vehicle_age * 100
                severity = max(500, sev_base * np.random.gamma(2, 0.5))
            else:
                severity = 0
            
            data.append({
                'age': age,
                'gender': gender,
                'territory': territory,
                'vehicle_age': vehicle_age,
                'num_claims': num_claims,
                'total_loss': round(severity * num_claims, 2),
            })
        
        # Train frequency model (Poisson)
        freq_result = sklearn_glm_tool.invoke({
            'model_name': 'test_auto_freq',
            'description': 'Auto claim frequency model',
            'data': data,
            'target_column': 'num_claims',
            'distribution': 'poisson',
            'categorical_columns': ['gender', 'territory'],
        })
        
        assert "TRAINING COMPLETE" in freq_result
        assert "POISSON" in freq_result
        
        # Train pure premium model (Tweedie)
        pp_result = sklearn_glm_tool.invoke({
            'model_name': 'test_auto_pp',
            'description': 'Auto pure premium model',
            'data': data,
            'target_column': 'total_loss',
            'distribution': 'tweedie',
            'tweedie_power': 1.5,
            'categorical_columns': ['gender', 'territory'],
        })
        
        assert "TRAINING COMPLETE" in pp_result
        assert "TWEEDIE" in pp_result
        
        # Verify both models are registered
        assert get_model_info("test_auto_freq") is not None
        assert get_model_info("test_auto_pp") is not None
        
        # Cleanup
        delete_model("test_auto_freq")
        delete_model("test_auto_pp")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

