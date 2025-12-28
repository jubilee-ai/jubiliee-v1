"""
Tests for Survival Analysis Training Tool

Tests Cox Proportional Hazards, Weibull AFT, and Log-Normal AFT models
for time-to-event prediction in insurance applications.
"""

import sys
import os

# Add training module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "models-tools", "training"))

import numpy as np
import pandas as pd
import pytest

from survival_analysis import SurvivalTrainingInput, train_survival_model, survival_analysis_tool
from model_storage import get_model_info, load_model, delete_model, predict_with_model_tool


class TestCoxProportionalHazards:
    """Tests for Cox PH model (semi-parametric survival)."""

    @pytest.fixture
    def policy_lapse_data(self):
        """Generate synthetic policy lapse data with censoring."""
        np.random.seed(42)
        data = []
        for _ in range(120):
            premium = int(np.random.uniform(500, 3000))
            age = int(np.random.uniform(25, 65))
            auto_pay = int(np.random.choice([0, 1], p=[0.4, 0.6]))
            num_policies = int(np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05]))
            
            # Hazard function
            hazard = 0.02
            hazard *= (1 + 0.0003 * premium)
            hazard *= (1 - 0.008 * age)
            hazard *= (0.5 if auto_pay else 1.5)
            hazard *= (1.5 if num_policies == 1 else 0.7)
            
            months_active = max(1, int(np.random.exponential(1 / max(hazard, 0.01))))
            
            # Censoring (30% still active)
            if np.random.random() < 0.3:
                lapsed = 0
                months_active = min(months_active, int(np.random.uniform(1, 24)))
            else:
                lapsed = 1
            
            data.append({
                'months_active': months_active,
                'lapsed': lapsed,
                'monthly_premium': premium,
                'customer_age': age,
                'auto_pay_enrolled': auto_pay,
                'num_policies': num_policies,
            })
        return data

    def test_cox_training(self, policy_lapse_data):
        """Test Cox Proportional Hazards training."""
        input_data = SurvivalTrainingInput(
            model_name="test_cox_lapse",
            description="Cox model for policy lapse",
            data=policy_lapse_data,
            duration_column="months_active",
            event_column="lapsed",
            model_type="cox",
            penalizer=0.01,
            random_state=42,
        )
        
        result = train_survival_model(input_data)
        
        assert result.success is True
        assert result.survival_model_type == "cox"
        assert result.n_samples == 120
        assert result.n_events > 0
        assert result.n_censored > 0
        assert 0.5 <= result.train_concordance <= 1.0
        assert len(result.coefficients) == 4
        
        # Check hazard ratios exist
        for feat, info in result.coefficients.items():
            assert "hazard_ratio" in info
            assert "p_value" in info
            assert info["hazard_ratio"] > 0
        
        delete_model("test_cox_lapse")

    def test_cox_tool_invocation(self, policy_lapse_data):
        """Test Cox model via LangChain tool."""
        result = survival_analysis_tool.invoke({
            'model_name': 'test_cox_tool',
            'description': 'Cox lapse via tool',
            'data': policy_lapse_data,
            'duration_column': 'months_active',
            'event_column': 'lapsed',
            'model_type': 'cox',
        })
        
        assert "COX PROPORTIONAL HAZARDS" in result
        assert "C-index" in result
        assert "HAZARD RATIOS" in result
        assert "MODEL REGISTERED" in result
        
        # Verify registration
        info = get_model_info("test_cox_tool")
        assert info is not None
        assert info["model_type"] == "survival_analysis"
        assert "concordance" in str(info["metrics"]).lower()
        
        delete_model("test_cox_tool")

    def test_cox_predictions(self, policy_lapse_data):
        """Test predictions with Cox model."""
        survival_analysis_tool.invoke({
            'model_name': 'test_cox_pred',
            'data': policy_lapse_data,
            'duration_column': 'months_active',
            'event_column': 'lapsed',
            'model_type': 'cox',
        })
        
        # Load and predict
        model = load_model("test_cox_pred")
        
        # High risk vs low risk profiles
        test_data = [
            {'monthly_premium': 2500, 'customer_age': 28, 'auto_pay_enrolled': 0, 'num_policies': 1},
            {'monthly_premium': 800, 'customer_age': 55, 'auto_pay_enrolled': 1, 'num_policies': 3},
        ]
        
        predictions = model.predict(pd.DataFrame(test_data))
        assert len(predictions) == 2
        assert predictions[0] > predictions[1]  # High risk has higher hazard
        
        delete_model("test_cox_pred")


class TestWeibullAFT:
    """Tests for Weibull Accelerated Failure Time model."""

    @pytest.fixture
    def time_to_claim_data(self):
        """Generate synthetic time-to-claim data."""
        np.random.seed(123)
        data = []
        for _ in range(100):
            driver_age = int(np.random.uniform(18, 70))
            years_licensed = max(0, driver_age - 18 + int(np.random.normal(0, 3)))
            vehicle_age = int(np.random.uniform(0, 12))
            annual_mileage = int(np.random.uniform(5000, 25000))
            
            # Risk factors affect time to claim
            base_time = 365
            risk_factor = 1.0
            risk_factor *= (0.6 if driver_age < 25 else 1.3 if driver_age > 55 else 1.0)
            risk_factor *= (0.7 if years_licensed < 3 else 1.1)
            risk_factor *= (0.85 if annual_mileage > 18000 else 1.0)
            
            days_to_claim = max(1, int(np.random.weibull(1.5) * base_time * risk_factor))
            
            # Censoring (35% haven't claimed yet)
            if np.random.random() < 0.35:
                claimed = 0
                days_to_claim = min(days_to_claim, int(np.random.uniform(30, 365)))
            else:
                claimed = 1
            
            data.append({
                'days_to_claim': days_to_claim,
                'claimed': claimed,
                'driver_age': driver_age,
                'years_licensed': years_licensed,
                'vehicle_age': vehicle_age,
                'annual_mileage': annual_mileage,
            })
        return data

    def test_weibull_training(self, time_to_claim_data):
        """Test Weibull AFT training."""
        input_data = SurvivalTrainingInput(
            model_name="test_weibull_claim",
            description="Weibull AFT for time to claim",
            data=time_to_claim_data,
            duration_column="days_to_claim",
            event_column="claimed",
            model_type="weibull",
            penalizer=0.01,
            random_state=42,
        )
        
        result = train_survival_model(input_data)
        
        assert result.success is True
        assert result.survival_model_type == "weibull"
        assert result.aic is not None  # Weibull has AIC
        assert result.bic is not None
        
        delete_model("test_weibull_claim")

    def test_weibull_predictions(self, time_to_claim_data):
        """Test Weibull predicts median survival time."""
        survival_analysis_tool.invoke({
            'model_name': 'test_weibull_pred',
            'data': time_to_claim_data,
            'duration_column': 'days_to_claim',
            'event_column': 'claimed',
            'model_type': 'weibull',
        })
        
        model = load_model("test_weibull_pred")
        
        # High risk vs low risk
        test_data = [
            {'driver_age': 21, 'years_licensed': 1, 'vehicle_age': 10, 'annual_mileage': 22000},
            {'driver_age': 55, 'years_licensed': 35, 'vehicle_age': 3, 'annual_mileage': 8000},
        ]
        
        predictions = model.predict(pd.DataFrame(test_data))
        assert len(predictions) == 2
        # Low risk should have longer survival time
        assert predictions[1] > predictions[0]
        
        delete_model("test_weibull_pred")


class TestLogNormalAFT:
    """Tests for Log-Normal AFT model."""

    @pytest.fixture
    def customer_tenure_data(self):
        """Generate synthetic customer tenure data."""
        np.random.seed(456)
        data = []
        for _ in range(90):
            acquisition_channel = np.random.choice(['direct', 'agent', 'online'])
            customer_segment = np.random.choice(['individual', 'family', 'business'])
            initial_premium = int(np.random.uniform(1000, 5000))
            
            # Tenure in months (lognormal-ish distribution)
            base_tenure = 24
            if acquisition_channel == 'agent':
                base_tenure *= 1.5
            if customer_segment == 'family':
                base_tenure *= 1.3
            
            tenure_months = max(1, int(np.random.lognormal(np.log(base_tenure), 0.5)))
            
            # Censoring
            if np.random.random() < 0.25:
                churned = 0
                tenure_months = min(tenure_months, int(np.random.uniform(3, 36)))
            else:
                churned = 1
            
            data.append({
                'tenure_months': tenure_months,
                'churned': churned,
                'acquisition_channel': acquisition_channel,
                'customer_segment': customer_segment,
                'initial_premium': initial_premium,
            })
        return data

    def test_lognormal_training(self, customer_tenure_data):
        """Test Log-Normal AFT training."""
        result = survival_analysis_tool.invoke({
            'model_name': 'test_lognormal_tenure',
            'description': 'Log-Normal AFT for customer tenure',
            'data': customer_tenure_data,
            'duration_column': 'tenure_months',
            'event_column': 'churned',
            'model_type': 'lognormal',
        })
        
        assert "LOG-NORMAL AFT" in result
        assert "TRAINING" in result or "MODEL REGISTERED" in result
        
        delete_model("test_lognormal_tenure")


class TestSurvivalValidation:
    """Tests for input validation and error handling."""

    def test_invalid_duration_fails(self):
        """Duration must be positive."""
        bad_data = [
            {'time': 0, 'event': 1, 'x': 1},
            {'time': 5, 'event': 0, 'x': 2},
        ]
        
        result = survival_analysis_tool.invoke({
            'model_name': 'test_bad_duration',
            'data': bad_data,
            'duration_column': 'time',
            'event_column': 'event',
        })
        
        assert "FAILED" in result or "positive" in result.lower()

    def test_invalid_event_values_fails(self):
        """Event column must be 0 or 1."""
        bad_data = [
            {'time': 10, 'event': 2, 'x': 1},
            {'time': 5, 'event': 1, 'x': 2},
        ]
        
        result = survival_analysis_tool.invoke({
            'model_name': 'test_bad_event',
            'data': bad_data,
            'duration_column': 'time',
            'event_column': 'event',
        })
        
        assert "FAILED" in result or "0" in result or "1" in result

    def test_missing_column_fails(self):
        """Should fail if columns don't exist."""
        data = [{'x': 1, 'y': 2}]
        
        result = survival_analysis_tool.invoke({
            'model_name': 'test_missing',
            'data': data,
            'duration_column': 'time',
            'event_column': 'event',
        })
        
        assert "FAILED" in result or "not found" in result.lower()


class TestSurvivalIntegration:
    """Integration tests for survival analysis workflow."""

    def test_full_mortality_workflow(self):
        """Test complete mortality modeling workflow."""
        np.random.seed(789)
        
        # Generate mortality data
        data = []
        for _ in range(150):
            entry_age = int(np.random.uniform(30, 70))
            gender = np.random.choice(['M', 'F'])
            smoker = int(np.random.choice([0, 1], p=[0.75, 0.25]))
            bmi = round(np.random.normal(26, 5), 1)
            
            # Mortality hazard
            base_hazard = 0.001
            base_hazard *= (1.02 ** (entry_age - 40))  # Age effect
            base_hazard *= (1.5 if gender == 'M' else 1.0)
            base_hazard *= (2.0 if smoker else 1.0)
            base_hazard *= (1.0 + 0.02 * max(0, bmi - 25))
            
            survival_years = max(1, int(np.random.exponential(1 / max(base_hazard, 0.0001))))
            
            # Censoring (policy still active)
            if np.random.random() < 0.4:
                death = 0
                survival_years = min(survival_years, int(np.random.uniform(1, 20)))
            else:
                death = 1
            
            data.append({
                'survival_years': survival_years,
                'death': death,
                'entry_age': entry_age,
                'gender': gender,
                'smoker': smoker,
                'bmi': bmi,
            })
        
        # Train Cox model
        cox_result = survival_analysis_tool.invoke({
            'model_name': 'test_mortality_cox',
            'description': 'Cox mortality model',
            'data': data,
            'duration_column': 'survival_years',
            'event_column': 'death',
            'model_type': 'cox',
        })
        
        assert "COX" in cox_result
        assert "C-index" in cox_result
        
        # Make predictions using predict tool
        test_profiles = [
            {'entry_age': 55, 'gender': 'M', 'smoker': 1, 'bmi': 32},  # High risk
            {'entry_age': 35, 'gender': 'F', 'smoker': 0, 'bmi': 22},  # Low risk
        ]
        
        predictions = predict_with_model_tool.invoke({
            'model_name': 'test_mortality_cox',
            'data': test_profiles,
        })
        
        assert "PREDICTIONS" in predictions
        assert "Row 1" in predictions
        assert "Row 2" in predictions
        
        delete_model("test_mortality_cox")

    def test_churn_prediction_workflow(self):
        """Test customer churn prediction with survival analysis."""
        np.random.seed(321)
        
        data = []
        for _ in range(100):
            tenure = max(1, int(np.random.exponential(24)))
            churned = 1 if np.random.random() > 0.3 else 0
            if not churned:
                tenure = min(tenure, int(np.random.uniform(1, 36)))
            
            data.append({
                'tenure_months': tenure,
                'churned': churned,
                'monthly_spend': int(np.random.uniform(50, 500)),
                'support_tickets': int(np.random.poisson(2)),
                'satisfaction_score': round(np.random.uniform(1, 5), 1),
            })
        
        result = survival_analysis_tool.invoke({
            'model_name': 'test_churn_survival',
            'data': data,
            'duration_column': 'tenure_months',
            'event_column': 'churned',
            'model_type': 'cox',
        })
        
        assert "MODEL REGISTERED" in result
        
        # Verify model works
        info = get_model_info("test_churn_survival")
        assert info is not None
        assert "concordance" in str(info["metrics"]).lower()
        
        delete_model("test_churn_survival")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

