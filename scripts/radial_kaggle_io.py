"""Explicit Kaggle transport for reviewed radial-study artifacts; no credentials saved."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def api(endpoint, payload=None):
    credentials = json.loads((Path.home() / ".kaggle/credentials.json").read_text())
    request = urllib.request.Request(
        "https://www.kaggle.com/api/v1/kernels/" + endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": "Bearer " + credentials["access_token"],
            "Content-Type": "application/json",
        },
    )

    # Authenticated calls must never forward a bearer token to a redirect host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise RuntimeError("Unexpected authenticated redirect")

    with urllib.request.build_opener(NoRedirect()).open(
        request, timeout=45
    ) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "download", "launch"))
    parser.add_argument("slug")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--internet", action="store_true")
    args = parser.parse_args()
    if not args.slug.startswith("ryangomez010/ripii-"):
        raise ValueError("Only this project's owner and notebook prefix are allowed")
    args.destination.mkdir(parents=True, exist_ok=True)
    owner, slug = args.slug.split("/", 1)
    query = "?" + urllib.parse.urlencode({"userName": owner, "kernelSlug": slug})
    if args.action == "launch":
        if (args.destination / "launch.json").exists():
            raise ValueError(
                "Launch receipt exists; inspect status, do not launch twice"
            )
        text = args.notebook.read_text()
        notebook = json.loads(text)
        assert notebook["nbformat"] == 4
        assert (
            notebook.get("metadata", {}).get("kernelspec", {}).get("name") == "python3"
        ), "Missing Python kernel metadata"
        request = {
            "slug": args.slug,
            "newTitle": slug.replace("-", " "),
            "text": text,
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": True,
            "enableGpu": not args.cpu,
            "enableInternet": args.internet,
            "sessionTimeoutSeconds": 1440,
            "datasetDataSources": [],
            "kernelDataSources": [],
            "competitionDataSources": [],
        }
        if not args.cpu:
            quota = api("quota")
            assert not quota["gpuQuota"]["isPayToScaleEnabled"], "Paid scaling enabled"
            request["machineShape"] = "NvidiaTeslaT4"
        response = api("push", request)
        receipt = {
            "response": response,
            "notebook_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "private": True,
            "gpu": not args.cpu,
            "internet": args.internet,
        }
        (args.destination / "launch.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(response))
        return
    status = api("status" + query)
    print(json.dumps(status))
    if args.action == "download" and status["status"] in (
        "complete",
        "error",
        "cancelAcknowledged",
    ):
        token = ""
        names = []
        while True:
            output = api(
                "output"
                + query
                + "&pageSize=100&pageToken="
                + urllib.parse.quote(token)
            )
            if output.get("log"):
                (args.destination / "api-log.json").write_text(output["log"])
            for item in output["files"]:
                name = item["fileName"]
                if "/" not in name and name.endswith("-results.zip"):
                    url = item["url"]
                    assert urllib.parse.urlparse(url).scheme == "https"
                    with (
                        urllib.request.urlopen(url, timeout=45) as response,
                        (args.destination / name).open("xb") as handle,
                    ):
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                    names.append(name)
            token = output.get("nextPageToken", "")
            if not token:
                break
        print(json.dumps({"downloaded": names}))


if __name__ == "__main__":
    main()
