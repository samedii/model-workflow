# MNIST problem

* Get datasets
* Metrics
    * Loss
    * Other
* Visualization of solution
* Web api (optional)
* High score?

## Usage

### Get datasets
Will return tensor dataset but could return a dataframe or other object.
If further custom pre-processing will be required per model then it is better
to have named data

    import mnist_problem
    ds = mnist_problem.get_dataset()

### Metrics
Example with distributions and log_prob

    features, labels = ds[:]
    mnist_problem.get_loss(labels, dist)

### Visualization
`model` behaves like a function and takes features as input to produce a distribution.

    mnist_problem.visualize(model)

Open `http://localhost:5000` in your browser

### Web api

    mnist_problem.web_api(model)

### High score

#### Submit
Creates `score.json` in `model_dir`

    mnist_problem.score_model(model, model_dir)

#### List
Produces list sorted by validation score

    python -m mnist_problem.highscore.ls parent/model/path
