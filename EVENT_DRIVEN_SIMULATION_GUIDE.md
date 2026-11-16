# Event-Driven Federated Learning: Simple Guide

## How the event-driven simulation works

Each client trains its own copy of the model. Clients can have different training speeds and network delays, so they do not all finish at the same time.

The simulation keeps a queue of events, such as:

- a client finishing its model download
- a client finishing a training time slice
- a client update arriving at the server

The simulation processes the next event in time order. When an update arrives, the server applies the notebook's update rule. That client can then download the latest server model and start another job. Virtual time advances according to the simulated speeds and delays; the jobs are not actually running in parallel on the computer.

## How buffering works

A buffer is a small waiting area at the server for client updates.

- **No buffer:** The server handles each update as soon as it arrives.
- **K=3 buffer:** The server waits until three updates have arrived, then aggregates those three together.

Buffering can make updates arrive at the server in groups. It does not make clients finish together; their arrival times can still differ.

## How staleness weighting works

A client trains using a copy of the server model from the time it downloaded that model. While the client trains, other clients may update the server. When the first client finally reports back, its update is then based on an older server version. The number of server updates since its download is called **staleness**.

The staleness-weighted notebooks reduce the influence of older updates. They use a factor of:

```text
1 / (1 + staleness)
```

For example, an update with staleness 0 keeps its full relative weight; at staleness 1 its factor is 1/2; at staleness 2 its factor is 1/3. In the buffered version, this factor is combined with the client's sample count when the three updates are aggregated.
