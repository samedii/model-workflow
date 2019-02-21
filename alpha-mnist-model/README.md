# Alpha MNIST model

* Config (perfect for hyperparameters and smaller changes)
* Pre-processing
* Model
    * Architecture and distribution (if NN)
* Custom training loop (optional)
* Model diagnostic metrics
* Custom problem metrics

## Usage

### Create

    python -m alpha_mnist_model.create runs/test

and modify `runs/test/config.json`

### Train

    python -m alpha_mnist_model.train runs/test

### Hyperparameter search

    for learning_rate in np.logspace(-5, 1, num=10):
        config = alpha_mnist_model.Config()
        config.learning_rate = float(learning_rate)
        config.save(f'runs/lr_{learning_rate:.1E}')
        alpha_mnist_model.train(config)

### Visualize
Uses `mnist_problem.visualize` to visualize solutions

### Web api
Uses `mnist_problem.web_api`

### Score model
Uses `mnist_problem.score_model`

    python -m alpha_mnist_model.score_model(model_dir, epoch)
