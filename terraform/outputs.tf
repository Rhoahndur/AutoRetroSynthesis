output "public_ip" {
  description = "Public IP of the GPU instance"
  value       = aws_eip.gpu.public_ip
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${aws_eip.gpu.public_ip}"
}

output "gradio_url" {
  description = "Gradio frontend URL"
  value       = "http://${aws_eip.gpu.public_ip}:7860"
}

output "instance_type" {
  description = "Instance type (for cost reference)"
  value       = var.instance_type
}

output "estimated_cost_per_hour" {
  description = "Estimated cost per hour"
  value       = "$0.526/hr (g4dn.xlarge on-demand)"
}
