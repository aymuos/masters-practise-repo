import requests
EXECUTE_FLAG = 'True'  # Set to 'True' to execute the code submission

if EXECUTE_FLAG == 'True':
    # Read code from the file,. replace test1.py with your file name
    with open("CH24M571.py", "r") as file:
        code_content = file.read()

    url = "https://lab.samsai.io/submit"
    headers = {
        "Authorization": "Bearer f9318293d4e405e5cff5d03a348a02ae0c4331916cd390041a700045d1bcb16a",
        "Content-Type": "application/json"
    }
    payload = {
        "code": code_content ,
        "student_id": "CH24M571",
    }

    response = requests.post(url, headers=headers, json=payload)

    print("Status Code:", response.status_code)
    print("Response Body:", response.text)