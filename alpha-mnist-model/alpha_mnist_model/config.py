import didactic_meme as model_suite


class Config(model_suite.Config):
    def default_values(self):
        return dict(
            learning_rate=1e-2,
            n_epochs=10,
            momentum=0.5,
            cuda=True,
            batch_size=64,
            eval_batch_size=1000,
            seed=123,
            log_interval=100,
        )
