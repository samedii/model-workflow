import didactic_meme as model_suite


class Config(model_suite.Config):
    def default_values(self):
        return dict(
            learning_rate=1e-3
        )
