import base64
import json

from flask import Flask, render_template, request

from usage_report import aggregate_threads


TABLE_HEADERS = [
    "user_id",
    "total_input_chars",
    "total_output_chars",
    "avg_turns",
    "total_minutes",
    "sessions",
    "code_gen",
    "file_attachments",
    "analysis_requests",
    "tool_usage",
    "days_used",
    "score",
    "rank",
    "paid_flag",
]

HEADER_LABELS = {
    "user_id": "ユーザーID",
    "total_input_chars": "総入力文字数",
    "total_output_chars": "総出力文字数",
    "avg_turns": "平均チャット往復数",
    "total_minutes": "合計利用時間（分）",
    "sessions": "セッション数",
    "code_gen": "コード生成回数",
    "file_attachments": "ファイル添付回数",
    "analysis_requests": "分析/要約依頼回数",
    "tool_usage": "ツール利用回数",
    "days_used": "直近30日間の使用日数",
    "score": "総合スコア",
    "rank": "評価ランク",
    "paid_flag": "有料版推奨",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024  # 32 MB


@app.route("/", methods=["GET", "POST"])
def index():
    report_df = None
    csv_data_uri = None
    errors = []

    if request.method == "POST":
        uploaded_files = request.files.getlist("files")
        payloads = []

        for uploaded in uploaded_files:
            if not uploaded or not uploaded.filename:
                continue

            try:
                raw_bytes = uploaded.read()
                if not raw_bytes:
                    errors.append(f"{uploaded.filename}: ファイルが空です。")
                    continue
                payload = json.loads(raw_bytes.decode("utf-8"))
                payloads.append((uploaded.filename, payload))
            except json.JSONDecodeError:
                errors.append(f"{uploaded.filename}: JSON の読み込みに失敗しました。")

        if not payloads and not errors:
            errors.append("JSON ファイルが選択されていません。")

        report_df = aggregate_threads(payloads)

        if not report_df.empty:
            csv_bytes = report_df.to_csv(index=False, encoding="utf-8-sig")
            csv_data_uri = (
                "data:text/csv;base64,"
                + base64.b64encode(csv_bytes.encode("utf-8-sig")).decode("ascii")
            )

    return render_template(
        "index.html",
        report_df=report_df,
        headers=TABLE_HEADERS,
        header_labels=HEADER_LABELS,
        csv_data_uri=csv_data_uri,
        errors=errors,
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
