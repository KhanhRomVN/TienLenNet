curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyBsfstqqic-87gu07dLW5R6Vjy3EDKoWo0" \
 -H 'Content-Type: application/json' \
 -X POST \
 -d '{
"contents": [
{
"parts": [
{
"text": "What is AI?"
}
]
}
]
}'
