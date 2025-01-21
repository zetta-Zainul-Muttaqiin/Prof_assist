## Flow after cloning:

1. Install venv
   `python -m venv venv`

2. activate venv:
   For macOS/Linux:
   `source venv/bin/activate`

   For Windows:
   `venv\Scripts\activate`

   for deactivate/exit venv
   `deactivate`

3. install all the dependencies from requirements.txt:
   `pip install -r requirements.txt`

4. Activate Virtual Environment (Venv) in the terminal project:
   If the project uses a virtual environment (often recommended to isolate dependencies), activate it:

   ```
   source venv/bin/activate  # On Unix or MacOS
   # or
   venv\Scripts\activate  # On Windows
   ```

5. To run the Flask:
   `flask --app app.py --debug run`

6. If want to install new package
   `pip install <package_name> && pip freeze >> requirements.txt`

## COMMAND

to install new dependencies:
`pip install <package_name> && pip freeze > requirements.txt`

to generate requirement.txt:
`pip freeze > requirements.txt`

To run the Flask with certain app (not automatically reload if made changes in code):
`python app.py`

what is venv and why use venv:
Venv is a Virtual Environment. It's good practice to use a virtual environment to isolate your project dependencies:
`python3 -m venv venv`

how to activate venv:
For macOS/Linux:
`source venv/bin/activate`

For Windows:
`venv\Scripts\activate`

for deactivate venv
`deactivate`
