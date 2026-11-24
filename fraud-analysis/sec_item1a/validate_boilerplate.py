
"""Fyntra - Manual validation of TF-IDF boilerplate detection.

Generate a reproducible, stratified review sample.

Sampling is based on company and predicted category.
Predictions are hidden from the main review worksheet
to reduce reviewer confirmation bias.
"""

import argparse
from pathlib import Path

import pandas as pd

from openpyxl.styles import (
    Font,
    PatternFill,
    Alignment
)

from openpyxl.worksheet.datavalidation import DataValidation


CATEGORIES = [
    "potential_boilerplate",
    "modified_disclosure",
    "potential_novel_disclosure"
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="./sec_item1a_data"
    )

    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.samples_per_category < 1:
        raise ValueError(
            "Samples per category must be positive."
        )

    root = Path(args.output_dir)

    source = (
        root / "item1a_pilot_boilerplate_tfidf.csv"
    )

    if not source.exists():
        raise FileNotFoundError(source)

    df = pd.read_csv(source)

    scored = df.loc[
        df["boilerplate_category"].isin(CATEGORIES)
        & df["max_similarity"].notna()
    ].copy()

    samples = []

    for (company, category), group in scored.groupby(
        ["company", "boilerplate_category"]
    ):

        sample_size = min(
            args.samples_per_category,
            len(group)
        )

        selected = group.sample(
            n=sample_size,
            random_state=args.seed
        )

        samples.append(selected)

    if not samples:
        raise ValueError("No review samples available.")

    review = pd.concat(
        samples,
        ignore_index=True
    )

    # Shuffle to hide category ordering.
    review = review.sample(
        frac=1,
        random_state=args.seed
    ).reset_index(drop=True)

    review.insert(
        0,
        "review_id",
        [
            f"BR_{i:03d}"
            for i in range(1, len(review) + 1)
        ]
    )

    review["manual_category"] = ""
    review["match_relevance"] = ""
    review["review_confidence"] = ""
    review["review_notes"] = ""

    review_columns = [
        "review_id",
        "company",
        "filing_date",
        "reference_filing_date",
        "chunk_id",
        "chunk_text",
        "matched_chunk_text",
        "manual_category",
        "match_relevance",
        "review_confidence",
        "review_notes"
    ]

    prediction_columns = [
        "review_id",
        "chunk_id",
        "matched_chunk_id",
        "max_similarity",
        "novelty_score",
        "boilerplate_category"
    ]

    instructions = pd.DataFrame({
        "Field": [
            "Purpose",
            "potential_boilerplate",
            "modified_disclosure",
            "potential_novel_disclosure",
            "uncertain",
            "match_relevance",
            "review_confidence",
            "review_notes",
            "Important limitation",
            "Review procedure"
        ],
        "Instructions": [
            "Assess current disclosure against the "
            "retrieved historical match.",

            "Substantially repeated wording and meaning "
            "with little meaningful change.",

            "Related disclosure with meaningful "
            "additions, removals or changes.",

            "Substantially different risk information "
            "relative to the retrieved match.",

            "Insufficient evidence, mixed content, "
            "or ambiguous classification.",

            "Choose high, partial, low or uncertain "
            "based on topical relevance.",

            "Choose high, medium or low confidence.",

            "Explain the classification and record "
            "formatting or segmentation problems.",

            "Novelty relative to one retrieved match "
            "does not prove novelty across the entire "
            "previous filing or indicate fraud.",

            "Complete the Review sheet before "
            "inspecting model predictions."
        ]
    })

    qa_dir = root / "qa"
    qa_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        qa_dir / "boilerplate_manual_review.xlsx"
    )

    # Prevent accidental loss of completed reviews.
    if output.exists():
        raise FileExistsError(
            f"Review workbook already exists: {output}. "
            "Rename or back up the existing workbook "
            "before regenerating."
        )

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        instructions.to_excel(
            writer,
            sheet_name="Instructions",
            index=False
        )

        review[review_columns].to_excel(
            writer,
            sheet_name="Review",
            index=False
        )

        review[prediction_columns].to_excel(
            writer,
            sheet_name="Model_Predictions",
            index=False
        )

        workbook = writer.book

        for sheet in workbook.worksheets:

            sheet.freeze_panes = "B2"

            sheet.auto_filter.ref = sheet.dimensions

            for cell in sheet[1]:

                cell.font = Font(
                    bold=True,
                    color="FFFFFF"
                )

                cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor="17365D"
                )

                cell.alignment = Alignment(
                    vertical="center",
                    wrap_text=True
                )

            sheet.row_dimensions[1].height = 32

            for row in sheet.iter_rows(min_row=2):

                for cell in row:

                    cell.alignment = Alignment(
                        vertical="top",
                        wrap_text=True
                    )

        instruction_sheet = workbook["Instructions"]

        instruction_sheet.column_dimensions["A"].width = 35
        instruction_sheet.column_dimensions["B"].width = 100

        for row_number in range(
            2,
            instruction_sheet.max_row + 1
        ):
            instruction_sheet.row_dimensions[
                row_number
            ].height = 45

        review_sheet = workbook["Review"]

        widths = {
            "A": 15,
            "B": 22,
            "C": 18,
            "D": 20,
            "E": 28,
            "F": 90,
            "G": 90,
            "H": 32,
            "I": 20,
            "J": 22,
            "K": 55
        }

        for column, width in widths.items():

            review_sheet.column_dimensions[
                column
            ].width = width

        for row_number in range(
            2,
            review_sheet.max_row + 1
        ):

            review_sheet.row_dimensions[
                row_number
            ].height = 210

            for column in ["H", "I", "J", "K"]:

                review_sheet[
                    f"{column}{row_number}"
                ].fill = PatternFill(
                    fill_type="solid",
                    fgColor="FFF2CC"
                )

        # Manual category dropdown.
        category_validation = DataValidation(
            type="list",
            formula1=(
                '"potential_boilerplate,'
                'modified_disclosure,'
                'potential_novel_disclosure,'
                'uncertain"'
            ),
            allow_blank=True
        )

        category_validation.showErrorMessage = True

        review_sheet.add_data_validation(
            category_validation
        )

        category_validation.add(
            f"H2:H{review_sheet.max_row}"
        )

        # Match relevance dropdown.
        relevance_validation = DataValidation(
            type="list",
            formula1='"high,partial,low,uncertain"',
            allow_blank=True
        )

        relevance_validation.showErrorMessage = True

        review_sheet.add_data_validation(
            relevance_validation
        )

        relevance_validation.add(
            f"I2:I{review_sheet.max_row}"
        )

        # Reviewer confidence dropdown.
        confidence_validation = DataValidation(
            type="list",
            formula1='"high,medium,low"',
            allow_blank=True
        )

        confidence_validation.showErrorMessage = True

        review_sheet.add_data_validation(
            confidence_validation
        )

        confidence_validation.add(
            f"J2:J{review_sheet.max_row}"
        )

        prediction_sheet = workbook[
            "Model_Predictions"
        ]

        for column in ["A", "B", "C", "D", "E", "F"]:

            prediction_sheet.column_dimensions[
                column
            ].width = 32

        # Hide predictions during manual review.
        prediction_sheet.sheet_state = "hidden"

        workbook.active = workbook.sheetnames.index(
            "Instructions"
        )

    print("=" * 60)
    print("FYNTRA - BOILERPLATE VALIDATION SAMPLE")
    print("=" * 60)

    print("Available scored chunks:", len(scored))
    print("Selected review samples:", len(review))
    print("Random seed:", args.seed)

    print("\nSamples by company and category:")

    print(
        pd.crosstab(
            review["company"],
            review["boilerplate_category"]
        ).to_string()
    )

    print("\nDuplicate review chunks:")
    print(review["chunk_id"].duplicated().sum())

    print("\nWorkbook saved:", output)

    print(
        "\nComplete the Review worksheet "
        "before viewing model predictions."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
