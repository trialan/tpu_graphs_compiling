from collections.abc import Sequence
import optuna
from absl import app
from tpu_graphs.baselines.layout import train_args
from tpu_graphs.baselines.layout import train_lib


def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    clip_norm = trial.suggest_float('clip_norm', 1e-5, 1e-2)

    # Get current training arguments and update them with suggested values
    args = train_args.get_args()
    updated_args = args._replace(learning_rate=learning_rate, clip_norm=clip_norm)

    # Run training with updated arguments
    val_opa = train_lib.train(updated_args)
    return val_opa

def main(unused_argv: Sequence[str]) -> None:
    study = optuna.create_study(direction='maximize')  # Adjust direction based on your metric
    study.optimize(objective, n_trials=60)

    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == '__main__':
    app.run(main)
