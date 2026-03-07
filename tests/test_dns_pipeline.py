import numpy as np
from utils.data_loader import load_and_preprocess_dns_data
from models.rf_evaluator import RFEvaluator

def test_data_loader_shapes():
    # Load a tiny fraction of data for swift testing (or trust the global data size)
    # Testing shapes is enough to ensure our loading pipeline is alive without running 10 minute downloads.
    
    # In a full test suite we'd mock the download, but here we just test end-to-end loading.
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names = load_and_preprocess_dns_data()
    
    # Test column consistency
    assert X_train.shape[1] == 32, f"Expected 32 features, got {X_train.shape[1]}"
    assert len(feature_names) == 32, f"Expected 32 feature names, got {len(feature_names)}"
    
    # Test Split Ratio: 50% test, 37.5% train, 12.5% val
    total_samples = X_test.shape[0] + X_train.shape[0] + X_val.shape[0]
    assert np.isclose(X_test.shape[0] / total_samples, 0.5, atol=0.01), "Test ratio is not 50%"
    assert np.isclose(X_train.shape[0] / total_samples, 0.375, atol=0.01), "Train ratio is not 37.5%"
    assert np.isclose(X_val.shape[0] / total_samples, 0.125, atol=0.01), "Val ratio is not 12.5%"

def test_rf_evaluator_health():
    # Create simple dummy tensor sizes mimicking the DNS dataset
    n_samples_train = 1000
    n_samples_val = 500
    n_features = 32
    
    X_train = np.random.rand(n_samples_train, n_features)
    y_train = np.random.randint(0, 4, n_samples_train)
    X_val = np.random.rand(n_samples_val, n_features)
    y_val = np.random.randint(0, 4, n_samples_val)
    
    evaluator = RFEvaluator(X_train, y_train, X_val, y_val)
    
    # Test Random Indices Evaluation
    random_indices = [0, 5, 12, 22, 31]
    f1_score = evaluator.get_fitness(random_indices)
    
    assert isinstance(f1_score, float), f"Fitness output was not a float, instead got {type(f1_score)}"
    assert 0.0 <= f1_score <= 1.0, f"F1 score {f1_score} violates bounds"

def test_rf_evaluator_penalty_logic():
    n_samples_train = 1000
    n_samples_val = 500
    n_features = 32
    
    X_train = np.random.rand(n_samples_train, n_features)
    y_train = np.random.randint(0, 4, n_samples_train)
    X_val = np.random.rand(n_samples_val, n_features)
    y_val = np.random.randint(0, 4, n_samples_val)
    
    evaluator = RFEvaluator(X_train, y_train, X_val, y_val)
    
    individual_zero = [0]*32
    individual_three = [1, 1, 1] + [0]*29
    
    # If 0 bits flipped, should instantly return 0
    f1_zero = evaluator.evaluate_with_penalty(individual_zero, 32)[0]
    assert f1_zero == 0.0, f"Individual with 0 bits got score {f1_zero}"
    
    f1_three = evaluator.evaluate_with_penalty(individual_three, 32)[0]
    assert isinstance(f1_three, float), "Evaluated penalty did not return a singular float in tuple"
