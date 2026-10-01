"""GET /model/info - which model is serving, how good it is, and the decision cut-offs."""
from fastapi import APIRouter, Depends

from app.auth import get_current_user
from app.models import User
from app.services.scoring import model_meta

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
def model_info(user: User = Depends(get_current_user)) -> dict:
    meta = model_meta()
    keys = ["model_name", "version", "model_type", "calibrated", "training_date", "features", "best_cv_auc",
            "n_train_val", "versions"]
    info = {k: meta[k] for k in keys}
    info["test_metrics"] = {k: v for k, v in meta["test"].items() if k != "confusion_matrix"}
    info["decision_cutoffs"] = {"review": meta["decision_cutoffs"]["review"],
                                "reject": meta["decision_cutoffs"]["reject"]}
    return info
