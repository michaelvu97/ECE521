import numpy as np
import matplotlib.pyplot as plt


for index in range(5):
    res = np.load("Results.1.4.1:" + str(index) + ".npy")
    """
    {
        "training_loss" : epoch_training_loss,
        "validation_loss" : epoch_validation_loss,
        "testing_loss": epoch_testing_loss,
        "training_error" : epoch_training_error,
        "validation_error" : epoch_validation_error,
        "testing_error": epoch_testing_error
    }
    """

    plt.plot(res.item().get("training_error"), label="Training")
    plt.plot(res.item().get("validation_error"), label="Validation")
    plt.plot(res.item().get("testing_error"), label="Testing")
    plt.legend()
    plt.title("Classification Error: Iteration " + str(index))
    plt.show()

    plt.plot(res.item().get("training_loss"), label="Training")
    plt.plot(res.item().get("validation_loss"), label="Validation")
    plt.plot(res.item().get("testing_loss"), label="Testing")
    plt.legend()
    plt.title("Classification Loss: Iteration " + str(index))
    plt.show()
