# Sample Legal Documents for JudicAIta

This directory contains sample legal documents for testing and demonstration purposes.

## Contents

### sample_contract.txt
A sample contract agreement demonstrating contract law concepts.

### sample_case_brief.txt
A sample case brief with citations and legal analysis.

## Usage

These files can be used with JudicAIta's document processing and analysis features:

```python
from judicaita.notebook_utils import NotebookHelper

helper = NotebookHelper()
result = helper.upload_and_analyze("examples/data/sample_case_brief.txt")
```

## Adding More Samples

To add your own sample documents:
1. Place PDF, DOCX, or TXT files in this directory
2. Update this README with descriptions
3. Ensure no sensitive or confidential information is included

## License

Sample documents in this directory are provided under the project's Apache 2.0 license
for educational and demonstration purposes only.
