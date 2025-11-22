# Triton Monitoring Stack

Complete production-grade monitoring for NVIDIA Triton Inference Server with YOLO models.

## Architecture

- **Prometheus**: Metrics collection and alerting (Triton + System metrics)
- **Node Exporter**: System-level CPU, memory, and hardware metrics
- **Grafana**: Visualization dashboards
- **Loki**: Log aggregation
- **Promtail**: Log shipping agent

## Quick Start

```bash
# Start all services including monitoring
docker compose up -d

# Check monitoring services
docker compose ps prometheus grafana loki promtail

# View logs
docker compose logs -f grafana
```

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |
| Loki | http://localhost:3100 | - |
| Triton Metrics | http://localhost:9502/metrics | - |

## Dashboard

### YOLO Triton Unified Dashboard

**Auto-loaded on startup** - Available immediately at http://localhost:3000

**The all-in-one production dashboard** combining model performance, GPU resources, and system metrics.

**Dashboard Layout:**

**Row 1: Service Overview (6 gauges)**
- Model Ready Status
- Total Throughput (req/s)
- Average Latency (ms)
- Queue Time (µs)
- Pending Requests
- GPU Usage (%)

**Row 2: System Overview (4 gauges)**
- System CPU Usage (%)
- System Memory Usage (%)
- System Load Average (1m)
- CPU Cores Count

**Row 3-6: Model Performance Analysis**
- Model Track Throughput Comparison (timeseries)
- Model Track Latency Comparison - Avg, P95, P99 (timeseries)
- Model Track Performance Leaderboard (bar gauge)
- Average Batch Size by Model Track
- DALI Preprocessing Batch Size (bottleneck indicator)

**Row 7-8: Queue & Errors**
- Queue Time Comparison (timeseries)
- Pending Requests by Model Track
- Failed Requests by Model Track

**Row 9-12: GPU Resources**
- GPU Utilization % (timeseries)
- GPU Memory Usage - Used vs Total (timeseries)
- GPU Power Usage (watts)
- GPU Temperature (celsius)

**Row 13-14: System Resources**
- CPU Utilization by Mode - Stacked % chart (user, system, iowait, idle)
- System Memory Breakdown - Used, cached, buffers, available

**Perfect for:**
- Single-pane-of-glass monitoring (like btop + nvidia-smi + inference metrics)
- Production operations
- Real-time troubleshooting
- Performance optimization
- Resource capacity planning

**Note:** Dashboard is auto-provisioned on startup. Manual import from `monitoring/dashboards/triton-unified-dashboard.json` is only needed for updates.

## Alerts

Alert rules are configured in `monitoring/alerts/triton-alerts.yml`

### Alert Thresholds

| Alert | Warning | Critical | Duration |
|-------|---------|----------|----------|
| Inference Latency | >100ms | >500ms | 2m / 1m |
| Failure Rate | >1% | >5% | 2m / 1m |
| GPU Utilization | >95% | - | 5m |
| GPU Memory | >90% | >95% | 5m / 2m |
| Queue Depth | >50 | - | 2m |
| Model Not Ready | - | Any | 1m |

### Viewing Active Alerts

1. **Prometheus**: http://localhost:9090/alerts
2. **Grafana**: Create alert panels in dashboards

## Logs with Loki

### Viewing Logs in Grafana

1. Go to **Explore** tab (compass icon)
2. Select **Loki** datasource
3. Use LogQL queries:

```logql
# All Triton logs
{container="/triton-api"}

# FastAPI logs
{container=~"/(yolo-api|pytorch-api)"}

# Error logs only
{container="/triton-api"} |= "error" or "ERROR"

# Inference logs
{container="/triton-api"} |= "inference"

# Last 5 minutes of warnings
{container="/triton-api"} |~ "warn|WARNING" [5m]
```

### Log Correlation

Combine logs with metrics in the same dashboard:
1. Add a Log panel using Loki datasource
2. Add metric panels using Prometheus
3. Sync time ranges to correlate events

## Key Metrics Reference

### Throughput Metrics

```promql
# Requests per second by model
rate(nv_inference_request_success[1m])

# Total throughput
sum(rate(nv_inference_request_success[1m]))
```

### Latency Metrics

```promql
# Average inference latency (ms)
rate(nv_inference_compute_infer_duration_us[1m])
  / rate(nv_inference_request_success[1m]) / 1000

# P95 latency
histogram_quantile(0.95,
  rate(nv_inference_request_duration_us_bucket[1m])) / 1000

# P99 latency
histogram_quantile(0.99,
  rate(nv_inference_request_duration_us_bucket[1m])) / 1000
```

### Batch Size Metrics

```promql
# Average batch size
rate(nv_inference_exec_count[1m])
  / rate(nv_inference_request_success[1m])
```

### GPU Metrics

```promql
# GPU utilization percentage (0-1, multiply by 100 for %)
nv_gpu_utilization * 100

# GPU memory used
nv_gpu_memory_used_bytes

# GPU memory percentage
(nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes) * 100
```

### System Metrics (Node Exporter)

```promql
# CPU utilization percentage
(1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))) * 100

# Memory utilization percentage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# CPU cores count
count(count(node_cpu_seconds_total{mode="idle"}) by (cpu))

# Load average
node_load1
node_load5
node_load15

# Memory breakdown
node_memory_MemTotal_bytes
node_memory_MemAvailable_bytes
node_memory_Cached_bytes
node_memory_Buffers_bytes
```

### Queue Metrics

```promql
# Pending requests
nv_inference_pending_request_count

# Average queue time (µs)
rate(nv_inference_queue_duration_us[1m])
  / rate(nv_inference_request_success[1m])
```

## Model Track Comparison Queries

### Compare throughput across tracks

```promql
# Show all YOLO model variants
rate(nv_inference_request_success{model=~"yolov11.*"}[1m])
```

### Find fastest model

```promql
# Lowest latency wins
topk(1,
  rate(nv_inference_compute_infer_duration_us{model=~"yolov11.*"}[5m])
  / rate(nv_inference_request_success{model=~"yolov11.*"}[5m]))
```

### Compare efficiency (throughput per GPU %)

```promql
# Requests per second per GPU utilization point
rate(nv_inference_request_success{model=~"yolov11.*"}[5m])
  / (nv_gpu_utilization / 100)
```

## Troubleshooting

### Prometheus not scraping

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify Triton metrics endpoint
curl http://localhost:9502/metrics
```

### Loki not receiving logs

```bash
# Check Promtail logs
docker compose logs promtail

# Verify Loki is ready
curl http://localhost:3100/ready

# Check Promtail targets
curl http://localhost:9080/targets
```

### Missing metrics in Grafana

1. Check datasource connection: **Configuration → Data Sources**
2. Verify query syntax in **Explore** tab
3. Check time range matches data availability
4. Ensure models are loaded and receiving requests

### Grafana shows "No data"

```bash
# Verify Prometheus is collecting metrics
curl http://localhost:9090/api/v1/query?query=up

# Check if Triton is exposing metrics
curl http://localhost:9502/metrics | grep nv_inference

# Restart Grafana to reload datasources
docker compose restart grafana
```

## Advanced Features

### Custom Dashboards

Create custom panels in Grafana:
1. **Time series**: Line/area charts for trends
2. **Stat**: Single value with thresholds
3. **Gauge**: Current value with min/max
4. **Bar gauge**: Compare multiple models
5. **Table**: Detailed breakdowns
6. **Logs**: Integrated log viewer

### Data Retention

Adjust retention in [docker-compose.yml](../docker-compose.yml):

```yaml
prometheus:
  command:
    - '--storage.tsdb.retention.time=30d'
    - '--storage.tsdb.retention.size=10GB'
```

### Performance Tips

1. **Use recording rules** for complex queries used in multiple dashboards
2. **Limit time ranges** to reduce query load
3. **Reduce scrape interval** if metrics aren't critical every 5s
4. **Set up remote write** for long-term storage (Thanos, Cortex, Mimir)

## Production Recommendations

For production deployments, consider:

- [ ] Set up Alertmanager for notifications (Slack, PagerDuty, email)
- [ ] Configure alert routing by severity
- [ ] Add remote write to long-term storage
- [ ] Enable authentication for external access
- [ ] Set up backup for Grafana dashboards
- [ ] Monitor the monitoring stack itself
- [ ] Document runbooks for common alerts
- [ ] Create SLO dashboards for business metrics
- [ ] Add multi-node monitoring if scaling horizontally

## File Structure

```
monitoring/
├── README.md                              # This file
├── prometheus.yml                         # Prometheus config (Triton + Node Exporter)
├── grafana-datasources.yml               # Grafana datasources (Prometheus + Loki)
├── grafana-dashboards.yml                # Dashboard auto-provisioning config
├── loki-config.yml                       # Loki log aggregation config
├── promtail-config.yml                   # Promtail log shipping config
├── alerts/
│   └── triton-alerts.yml                 # Prometheus alert rules
└── dashboards/
    └── triton-unified-dashboard.json     # All-in-one unified dashboard (auto-loaded)
```

## Available Metrics

### Triton Inference Metrics

Exposed by Triton on port 8002 (mapped to 9502):

| Metric | Description |
|--------|-------------|
| `nv_inference_request_success` | Total successful requests |
| `nv_inference_request_failure` | Total failed requests |
| `nv_inference_count` | Number of inferences performed |
| `nv_inference_exec_count` | Number of model executions |
| `nv_inference_request_duration_us` | End-to-end request duration |
| `nv_inference_queue_duration_us` | Time in queue |
| `nv_inference_compute_input_duration_us` | Input processing time |
| `nv_inference_compute_infer_duration_us` | Inference computation time |
| `nv_inference_compute_output_duration_us` | Output processing time |
| `nv_inference_pending_request_count` | Current queue depth |
| `nv_model_ready_state` | Model availability (1=ready) |
| `nv_gpu_utilization` | GPU usage (0-1 decimal, multiply by 100 for %) |
| `nv_gpu_memory_total_bytes` | Total GPU memory |
| `nv_gpu_memory_used_bytes` | Used GPU memory |
| `nv_gpu_power_usage` | GPU power consumption (watts) |
| `nv_gpu_power_limit` | GPU power limit (watts) |
| `nv_gpu_temperature` | GPU temperature (celsius) |
| `nv_energy_consumption` | Total energy consumed |

**View all Triton metrics:** `curl http://localhost:9502/metrics`

### System Metrics (Node Exporter)

Exposed on port 9100:

| Metric | Description |
|--------|-------------|
| `node_cpu_seconds_total` | CPU time by mode (user, system, idle, iowait, etc.) |
| `node_memory_MemTotal_bytes` | Total system memory |
| `node_memory_MemAvailable_bytes` | Available system memory |
| `node_memory_Cached_bytes` | Cached memory |
| `node_memory_Buffers_bytes` | Buffer memory |
| `node_load1` | 1-minute load average |
| `node_load5` | 5-minute load average |
| `node_load15` | 15-minute load average |
| `node_disk_*` | Disk I/O metrics |
| `node_network_*` | Network interface metrics |

**View all system metrics:** `curl http://localhost:9100/metrics`
