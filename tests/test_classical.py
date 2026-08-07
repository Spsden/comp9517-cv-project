import numpy as np
import torch

from inat_project.classical import (
    evaluate_linear_svc,
    extract_hog,
    extract_hsv,
    tune_sgd_hinge,
    tune_linear_svc,
)


def test_handcrafted_feature_dimensions():
    image = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
    hog_features = extract_hog(image)
    hsv_features = extract_hsv(image)

    assert hog_features.shape == (8100,)
    assert hsv_features.shape == (64,)
    assert hog_features.dtype == np.float32
    assert hsv_features.dtype == np.float32
    assert np.isclose(hsv_features.sum(), 1.0)


def test_linear_svc_tuning_and_common_metrics():
    generator = np.random.default_rng(42)
    x_train = generator.normal(size=(24, 8)).astype(np.float32)
    y_train = np.repeat(np.arange(3), 8)
    x_validation = generator.normal(size=(12, 8)).astype(np.float32)
    y_validation = np.repeat(np.arange(3), 4)

    model, tuning = tune_linear_svc(
        x_train,
        y_train,
        x_validation,
        y_validation,
        c_values=[0.01, 0.1],
        max_iter=1000,
    )
    predictions, metrics = evaluate_linear_svc(
        model,
        x_validation,
        y_validation,
        [f"image_{index}.jpg" for index in range(len(y_validation))],
        num_classes=3,
    )

    assert len(tuning) == 2
    assert len(predictions) == len(y_validation)
    assert metrics["test_images"] == len(y_validation)
    assert metrics["top5_accuracy"] == 1.0


def test_sgd_hinge_tuning_uses_common_model_interface():
    generator = np.random.default_rng(7)
    x_train = generator.normal(size=(30, 10)).astype(np.float32)
    y_train = np.repeat(np.arange(3), 10)
    x_validation = generator.normal(size=(12, 10)).astype(np.float32)
    y_validation = np.repeat(np.arange(3), 4)

    model, tuning = tune_sgd_hinge(
        x_train,
        y_train,
        x_validation,
        y_validation,
        c_values=[0.01, 0.1],
        epochs=2,
    )

    assert len(tuning) == 2
    assert model.predict(x_validation).shape == (len(x_validation),)
    assert model.decision_function(x_validation).shape == (len(x_validation), 3)
