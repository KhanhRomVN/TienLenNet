curl -X POST https://server.trypear.ai/pearai-server-api2/v1/server_chat \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsImtpZCI6IlpWRlZJa0g5Z0EyVjJMbFciLCJ0eXAiOiJKV1QifQ.eyJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJvYXV0aCIsInRpbWVzdGFtcCI6MTc1MzAxNTE4OH1dLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJnb29nbGUiLCJwcm92aWRlcnMiOlsiZ29vZ2xlIl19LCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZW1haWwiOiJ0b3JpdmVsYXNxdWV6dmltMjI2NzFrQGNtcDgub3JnIiwiZXhwIjoxNzUzMTAxNTkwLCJpYXQiOjE3NTMwMTUxOTAsImlzX2Fub255bW91cyI6ZmFsc2UsImlzcyI6Imh0dHBzOi8vd21xd3h4anBqcGhic3BrY3h0anQuc3VwYWJhc2UuY28vYXV0aC92MSIsInBob25lIjoiIiwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJzZXNzaW9uX2lkIjoiODQ3YzMxNDktNmJkZi00OTg2LWJlNzktZTA4NDZkYmNhY2NhIiwic3ViIjoiNDFmMmE0ZTctZmJlYS00YzdmLTliMjQtNmJjMDgwYWEwZjIwIiwic3Vic2NyaXB0aW9uX2N1c3RvbWVyX2lkIjpudWxsLCJzdWJzY3JpcHRpb25fcGVyaW9kX2VuZCI6bnVsbCwic3Vic2NyaXB0aW9uX3ByaWNpbmdfdGllciI6ImZyZWUiLCJ1c2VyX21ldGFkYXRhIjp7ImF2YXRhcl91cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NMSkh5aWw0NjMyblFoa01MQmVlNnFEZTFtREdWdC1TQll3Y0o2NWJROGJja3YtRWc9czk2LWMiLCJjdXN0b21fY2xhaW1zIjp7ImhkIjoiY21wOC5vcmcifSwiZW1haWwiOiJ0b3JpdmVsYXNxdWV6dmltMjI2NzFrQGNtcDgub3JnIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZ1bGxfbmFtZSI6IkNsYXJldGEgVmVsYXNxdWV6IiwiaXNzIjoiaHR0cHM6Ly9hY2NvdW50cy5nb29nbGUuY29tIiwibmFtZSI6IkNsYXJldGEgVmVsYXNxdWV6IiwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jTEpIeWlsNDYzMm5RaGtNTEJlZTZxRGUxbURHVnQtU0JZd2NKNjViUThiY2t2LUVnPXM5Ni1jIiwicHJvdmlkZXJfaWQiOiIxMTI0NzUxNTQ2MTg0NDE0NTY4MTEiLCJzdWIiOiIxMTI0NzUxNTQ2MTg0NDE0NTY4MTEifX0.aP8pjjo8JDZofVNkVutHRWwU4_s8bNiPSRK8EcTyfLM" \
 -d '{
"messages": [
{
"role": "user",
"content": [
{
"type": "text",
"text": "hello"
}
]
},
{
"role": "assistant",
"content": [
{
"type": "text",
"text": "Hey there! 👋😊 How can I help you today?"
}
]
},
{
"role": "user",
"content": [
{
"type": "text",
"text": "xin chào"
}
]
}
],
"title": "PearAI Model (Recommended)",
"completionOptions": {}
}'
