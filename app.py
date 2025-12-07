from flask import Flask, request, jsonify, send_file
from flask_cors import CORS 
import numpy as np
import torch
import os
from model_utils import load_model, get_model_latent, get_activity_latent
from site_information import forecast_sites

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your PyTorch model
data_dir = "models/vfhnn_beta_1"
sites_rmse_cache = {}  # simple in-memory cache: site_str -> rmse_list
forecast_cache = {} 
decoder, config, device = load_model(data_dir, device)

@app.route("/")
def home():
    return "Hello from Fly.io!"

@app.route("/api/latent_activity", methods=["POST"])
def latent_activity():
    mu, logvar = get_model_latent(forecast_sites, data_dir)  # shape: (num_layers, ...)
    var = np.exp(logvar)  # convert logvar to actual variance

    # activity can still be variance across time/dim
    activity = get_activity_latent(mu)

    # Prepare data for frontend traversal: send mean and std per latent
    # For simplicity, we can return mean and std per latent dimension for each layer
    # Here we collapse time dimension if needed (take mean over time axis 0)
    latent_info = []
    for layer_idx in range(mu.shape[0]):
        layer_means = mu[layer_idx].mean(axis=0).tolist()       # mean across time
        layer_stds = np.sqrt(var[layer_idx].mean(axis=0)).tolist()  # std across time
        latent_info.append({"mean": layer_means, "std": layer_stds})

    return jsonify({
        "activity": activity.tolist(),
        "latent_info": latent_info  # send mean/std for slider traversal
    })

@app.route("/api/site_rmse_image", methods=["GET"])
def site_rmse_image():
    """
    Returns the precomputed per-day RMSE/bias PNG for the requested site if available.
    Query param: ?site=1436
    """
    site = request.args.get("site")
    if not site:
        return jsonify({"error": "missing site parameter"}), 400

    # data_dir variable is defined earlier in this file as "models/vfhnn_beta_1"
    img_name = f"{site}_per_day_rmse_bias.png"
    img_path = os.path.join(data_dir, img_name)
    if os.path.exists(img_path):
        return send_file(img_path, mimetype="image/png")
    else:
        # fallback: try prediction plot or any other available visual
        alt_name = f"{site}_per_day_prediction_plot.png"
        alt_path = os.path.join(data_dir, alt_name)
        if os.path.exists(alt_path):
            return send_file(alt_path, mimetype="image/png")
        return jsonify({"error": "rmse image not found for site"}), 404

@app.route("/api/sites_rmse", methods=["GET"])
def sites_rmse():
    """
    Returns per-lead-day RMSE arrays for one or more sites.
    Query param: ?sites=1436,1449  (comma-separated)
    Response:
    {
        "sites_rmse": {
            "1436": [rmse_day1, rmse_day2, ...],
            "1449": [...]
        },
        "horizon": H
    }
    """
    sites_param = request.args.get("sites")
    if not sites_param:
        return jsonify({"error": "missing sites parameter (comma separated)"}), 400

    # Accept either comma-separated numbers or single site
    sites = [s.strip() for s in sites_param.split(",") if s.strip()]
    if not sites:
        return jsonify({"error": "no valid sites provided"}), 400

    def compute_metrics(pred, target):
        # Align and normalize arrays similar to your notebook's compute_metrics
        pred = np.asarray(pred)
        target = np.asarray(target)

        # Collapse trailing channel dimension if present (e.g., (N,H,C) -> mean over C)
        if pred.ndim == 3:
            pred = pred.mean(axis=-1)
        if target.ndim == 3:
            target = target.mean(axis=-1)

        # If 1D (N,), make it (N,1)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if target.ndim == 1:
            target = target.reshape(-1, 1)

        # Ensure at least 2D now
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Unsupported array dims after normalization: pred.ndim={pred.ndim}, target.ndim={target.ndim}")

        # Align sample counts
        min_samples = min(pred.shape[0], target.shape[0])
        if min_samples == 0:
            return None, None
        pred = pred[:min_samples, :]
        target = target[:min_samples, :]

        # Align horizon (number of lead days)
        min_h = min(pred.shape[1], target.shape[1])
        if min_h == 0:
            return None, None
        pred = pred[:, :min_h]
        target = target[:, :min_h]

        # bias and rmse per lead day
        bias = np.mean(pred - target, axis=0)
        rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))
        return bias, rmse

    # Candidates for keys (prioritize exact names from your notebook)
    pred_keys_priority = ["predictions", "pred", "forecast", "y_pred", "predictions_array", "outputs"]
    target_keys_priority = ["target", "obs", "y_true", "y", "target_array", "observations"]

    sites_rmse = {}
    min_horizon = None

    for site in sites:
        try:
            site_str = str(site)

            # Check cache first
            if site_str in sites_rmse_cache:
                cached = sites_rmse_cache[site_str]
                sites_rmse[site_str] = cached
                # update min_horizon
                h = len(cached) if isinstance(cached, list) else None
                if h is not None:
                    min_horizon = h if min_horizon is None else min(min_horizon, h)
                continue

            npz_path = os.path.join(data_dir, f"{site_str}_forecast_outputs.npz")
            if not os.path.exists(npz_path):
                sites_rmse[site_str] = {"error": "forecast_outputs file not found"}
                continue

            npz = np.load(npz_path, allow_pickle=True)
            keys = list(npz.files)

            # helper to find key by priority list
            def pick_key(priority_list):
                for name in priority_list:
                    for k in keys:
                        if k.lower() == name.lower():
                            return k
                # try partial match if exact not found
                for name in priority_list:
                    for k in keys:
                        if name.lower() in k.lower():
                            return k
                return None

            pred_key = pick_key(pred_keys_priority)
            target_key = pick_key(target_keys_priority)

            # fallback: if keys not found, try common names or first two arrays
            if pred_key is None or target_key is None:
                if "predictions" in keys and "target" in keys:
                    pred_key = pred_key or "predictions"
                    target_key = target_key or "target"
                elif len(keys) >= 2:
                    pred_key = pred_key or keys[0]
                    target_key = target_key or keys[1]
                else:
                    sites_rmse[site_str] = {"error": f"could not find predictions/target arrays (keys: {keys})"}
                    continue

            y_pred = npz[pred_key]
            y_true = npz[target_key]

            # compute metrics using the notebook-style compute_metrics
            bias, rmse = compute_metrics(y_pred, y_true)
            if rmse is None:
                sites_rmse[site_str] = {"error": "no data after alignment"}
                continue

            # convert to python lists and cache
            rmse_list = rmse.tolist()
            sites_rmse[site_str] = rmse_list
            sites_rmse_cache[site_str] = rmse_list  # cache the result

            # track minimal horizon
            h = len(rmse_list)
            if min_horizon is None:
                min_horizon = h
            else:
                min_horizon = min(min_horizon, h)

        except Exception as e:
            sites_rmse[str(site)] = {"error": f"exception while loading: {str(e)}"}

    # If horizons differ among sites, trim rmse lists to the minimal horizon
    if min_horizon is not None:
        for s, v in list(sites_rmse.items()):
            if isinstance(v, list) and len(v) > min_horizon:
                sites_rmse[s] = v[:min_horizon]

    return jsonify({"sites_rmse": sites_rmse, "horizon": min_horizon})

@app.route("/api/site_forecast", methods=["GET"])
def site_forecast():
    """
    Return predictions and target arrays for a specific site and lead day.
    Query params:
        site: site id (e.g. 1436)
        lead: lead day (1-based). If omitted, returns full arrays for all leads.
    Response JSON:
        {
            "site": "1436",
            "lead": 1,
            "pred": [...],       # predictions for the requested lead (list) OR nested list if full
            "target": [...],
            "horizon": H
        }
    """
    site = request.args.get("site")
    lead = request.args.get("lead", None)  # optional, 1-based

    if not site:
        return jsonify({"error": "missing site parameter"}), 400

    site_str = str(site)

    try:
        # If not cached, load NPZ
        if site_str not in forecast_cache:
            npz_path = os.path.join(data_dir, f"{site_str}_forecast_outputs.npz")
            if not os.path.exists(npz_path):
                return jsonify({"error": "forecast_outputs file not found for site"}), 404

            npz = np.load(npz_path, allow_pickle=True)
            keys = list(npz.files)

            # Prefer 'predictions' and 'target' keys (per your notebook)
            pred_key = None
            target_key = None
            for k in keys:
                kl = k.lower()
                if pred_key is None and ("prediction" in kl or "pred" == kl or "predictions" in kl or "outputs" in kl):
                    pred_key = k
                if target_key is None and ("target" in kl or "obs" in kl or "y_true" in kl or "y" == kl):
                    target_key = k

            # fallback to first two keys if not found
            if pred_key is None or target_key is None:
                if len(keys) >= 2:
                    pred_key = pred_key or keys[0]
                    target_key = target_key or keys[1]
                else:
                    return jsonify({"error": f"could not find prediction/target arrays keys in {npz_path}, keys={keys}"}), 500

            y_pred = np.asarray(npz[pred_key])
            y_true = np.asarray(npz[target_key])

            # Normalize shapes:
            # If 3D (N, H, C) -> average over channels -> (N, H)
            if y_pred.ndim == 3:
                y_pred = y_pred.mean(axis=-1)
            if y_true.ndim == 3:
                y_true = y_true.mean(axis=-1)

            # Ensure at least 2D
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)

            # align sample counts and horizon by trimming to minimums
            min_samples = min(y_pred.shape[0], y_true.shape[0])
            min_horizon = min(y_pred.shape[1], y_true.shape[1])
            y_pred = y_pred[:min_samples, :min_horizon]
            y_true = y_true[:min_samples, :min_horizon]

            # cache arrays (store as lists here? Keep as numpy for quick slicing)
            forecast_cache[site_str] = {"predictions": y_pred, "target": y_true}

        # retrieve cached arrays
        cached = forecast_cache[site_str]
        y_pred = cached["predictions"]
        y_true = cached["target"]

        horizon = y_pred.shape[1]

        # If a particular lead requested, return vectors for that lead (1-based)
        if lead is not None:
            try:
                lead_i = int(lead) - 1
            # fallback if lead not integer
            except:
                return jsonify({"error": "invalid lead parameter"}), 400
            if lead_i < 0 or lead_i >= horizon:
                return jsonify({"error": "lead out of range", "horizon": horizon}), 400

            pred_vec = y_pred[:, lead_i].tolist()
            target_vec = y_true[:, lead_i].tolist()
            return jsonify({"site": site_str, "lead": lead_i + 1, "pred": pred_vec, "target": target_vec, "horizon": horizon})

        # otherwise return full arrays (careful: may be larger)
        return jsonify({"site": site_str, "lead": None, "pred": y_pred.tolist(), "target": y_true.tolist(), "horizon": horizon})

    except Exception as e:
        return jsonify({"error": f"exception while loading site forecast: {str(e)}"}), 500

@app.route("/api/forecast_sites", methods=["GET"])
def get_forecast_sites():
    return jsonify({"sites": forecast_sites})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
