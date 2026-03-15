terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# Find latest Deep Learning AMI (Ubuntu)
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Security group
resource "aws_security_group" "autoresearch" {
  name        = "autoresearch-retro"
  description = "SSH + Gradio access for autoresearch"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  ingress {
    description = "Gradio"
    from_port   = 7860
    to_port     = 7860
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 instance
resource "aws_instance" "gpu" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.instance_type
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.autoresearch.id]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    cd /home/ubuntu

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> /home/ubuntu/.bashrc
    export PATH="/home/ubuntu/.local/bin:$PATH"

    # Clone repo
    git clone ${var.repo_url} autoresearch-retro
    cd autoresearch-retro

    # Install dependencies and prepare data
    uv sync
    uv run prepare.py

    # Signal ready
    touch /home/ubuntu/READY
    echo "Setup complete at $(date)" >> /home/ubuntu/setup.log
  EOF

  tags = {
    Name = "autoresearch-retro"
  }
}

# Elastic IP
resource "aws_eip" "gpu" {
  instance = aws_instance.gpu.id
  domain   = "vpc"
}
