import requests
import json
import pandas as pd

url = "http://localhost:3003/v2/products/update"

review_df = pd.read_csv("data_to_update.csv")

def update_product(row):
  _id = row["ID"]
  _name = row["Name"]
  _description = row["Description"]
  _specifications = row["Specifications"]

  payload = json.dumps({
    "id": _id,
    "name": str(_name),
    "description": str(_description),
    "specifications": str(_specifications)
  })
  headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Miwicm9sZSI6ImFkbWluIiwiaWF0IjoxNzIzODAxNzk5LCJleHAiOjE3MjM4MTI1OTl9.TL40un6iU-P5RUjFMKzh0AjTpNnkm_QYaPS_EHHdpx4'
  }

  response = requests.request("PUT", url, headers=headers, data=payload)
  print(response.text)

review_df.apply(update_product, axis=1)


