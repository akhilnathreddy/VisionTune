from collections import Counter

from conftest import make_image, sqa_row
from visiontune.data.prepare import NEAR_DUP_MAX_HAMMING, build_split, prepare_scienceqa
from visiontune.data.records import (hamming, image_dhash, scienceqa_record, trim_explanation,
                                     validate_record)
from visiontune.data.sampling import select_subset
from visiontune.prompts import build_messages, build_target, build_user_text


def test_trim_explanation_cuts_at_sentence():
    text = "First sentence here. Second sentence is longer than the limit allows."
    assert trim_explanation(text, 30) == "First sentence here."
    assert trim_explanation("short", 30) == "short"


def test_dhash_is_stable_under_resize_and_jpeg():
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 70, 60), fill="navy")
    draw.ellipse((90, 40, 150, 110), fill="darkred")
    draw.line((0, 119, 159, 0), fill="black", width=3)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    rejpeg = Image.open(buf)
    h = image_dhash(img)
    assert hamming(h, image_dhash(img.resize((320, 240)))) <= NEAR_DUP_MAX_HAMMING
    assert hamming(h, image_dhash(rejpeg)) <= NEAR_DUP_MAX_HAMMING
    assert hamming(h, image_dhash(make_image(3, size=(160, 120)))) > 16


def test_validation_reasons():
    rec = scienceqa_record(sqa_row(), "train", 0, make_image(0))
    assert validate_record(rec, require_image=True) is None
    no_img = scienceqa_record(sqa_row(image_seed=None), "train", 1, None)
    assert validate_record(no_img, require_image=True) == "no_image"
    assert validate_record(no_img, require_image=False) is None
    bad = dict(rec, answer_index=5)
    assert validate_record(bad, True) == "answer_out_of_range"


def test_build_split_drops_duplicates_and_text_only():
    rows = [sqa_row(image_seed=1), sqa_row(image_seed=1), sqa_row(image_seed=None), sqa_row(image_seed=2)]
    recs, reasons = build_split(rows, "train", scienceqa_record, require_image=True)
    assert [r["hf_index"] for r in recs] == [0, 3]
    assert reasons == Counter({"duplicate_in_split": 1, "no_image": 1})


def test_prepare_removes_cross_split_leakage(base_cfg):
    splits = {
        "train": [sqa_row(image_seed=1), sqa_row(image_seed=2), sqa_row(question="Q3?", image_seed=3)],
        "validation": [sqa_row(image_seed=2), sqa_row(question="Q5?", image_seed=5)],
        "test": [sqa_row(image_seed=1), sqa_row(question="Q6?", image_seed=6)],
    }
    out, report = prepare_scienceqa(base_cfg, load_split=lambda _id, split, _rev=None: splits[split])
    assert [r["question"] for r in out["train"]] == ["Q3?"]
    assert report["splits"]["train"]["excluded"] == {"exact_dup_of_test": 1, "exact_dup_of_validation": 1}
    assert len(out["validation"]) == 2 and len(out["test"]) == 2


def _near_dup_splits():
    test_row = sqa_row(question="Q?", image_seed=9)
    resized = dict(test_row, image=test_row["image"].resize((128, 96)))   # same figure, new pixels
    other = sqa_row(question="Q?", image_seed=10)                       # same text, different figure
    novel = sqa_row(question="Novel?", image_seed=11)
    return {"train": [resized, other], "validation": [], "test": [test_row, novel]}


def test_near_duplicates_are_flagged_by_default(base_cfg):
    splits = _near_dup_splits()
    out, report = prepare_scienceqa(base_cfg, load_split=lambda _id, split, _rev=None: splits[split])
    assert len(out["train"]) == 2
    assert [r["near_dup_in_train"] for r in out["test"]] == [True, False]
    assert report["splits"]["test"]["near_dup_in_train"] == {"flagged": 1, "same_answer": 1}


def test_near_duplicates_can_be_excluded(base_cfg):
    base_cfg["data"]["near_dup_policy"] = "exclude"
    splits = _near_dup_splits()
    out, report = prepare_scienceqa(base_cfg, load_split=lambda _id, split, _rev=None: splits[split])
    assert [r["hf_index"] for r in out["train"]] == [1]
    assert report["splits"]["train"]["excluded"] == {"near_dup_of_test": 1}


def test_subsets_are_nested_and_deterministic():
    recs = [{"id": f"r{i}"} for i in range(100)]
    small, big = select_subset(recs, 10, 42), select_subset(recs, 40, 42)
    assert [r["id"] for r in small] == [r["id"] for r in big[:10]]
    assert select_subset(recs, 10, 42) == small and select_subset(recs, 10, 7) != small
    assert len(select_subset(recs, None, 1)) == 100


def test_prompt_and_target_format():
    rec = scienceqa_record(sqa_row(hint="Look closely."), "train", 0, make_image(0))
    user = build_user_text(rec)
    assert user.startswith("Context: Look closely.\nQuestion: Which is a mammal?")
    assert "A. whale\nB. shark" in user
    assert build_target(rec) == "Answer: A. whale\nExplanation: Whales breathe air. Sharks have gills."
    msgs = build_messages(rec, "sys", with_answer=True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"][0] == {"type": "image"}
