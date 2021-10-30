import requests

url = 'http://127.0.0.1:5000/results'
r = requests.post(url,json={'gender':0, 'senior':0, 'partner':1, 'dependent':0,
                            'tenure':28, 'phone':1, 'multiple_lines':1, 'internet':1,
                            'security':0, 'backup':0, 'protection':1, 'support':1,
                            'tv':1, 'movies':1, 'paperless':1, 'payment_method':0,
                            'monthly_charges':104.8, 'loyalty':0})

print(r.json())
