"""Frame-conserving plans on H3's aligned window/context lengths."""
def window_plan(total_frames, window_frames, context_frames, seed):
    size, overlap = int(window_frames), int(context_frames)
    if overlap < 0:
        raise ValueError('Context frames cannot be negative')
    if size <= overlap:
        raise ValueError('The H3 window must leave frames after its pinned context')
    entries = []
    retained = 0
    while retained < total_frames:
        index = len(entries)
        trim = overlap if index else 0
        start = retained - trim
        end = min(total_frames, start + size)
        entries.append({'index': index, 'source_start': start, 'source_end_exclusive': end,
                        'source_frames': total_frames, 'window_frames': size,
                        'history_frames': overlap, 'discarded_overlap': trim, 'seed': seed})
        retained = end
    return entries
