r'''longctx.py -- 长上下文问答路径（分块 prefill，8GB 卡可跑 32k）
================================================================================
背景: tower_runtime 原本是纯查表解码(无生成路径)，因此没有长上下文能力。
本模块补上这条路径，并复用塔的记忆：
    use_memory=True  先用塔查表(§6.7 零训练路线)，命中则把答案作为一行短记忆前置
                     （§4.6 证明『检索+文本』在所有底座上都成立，且只要约 30 token）
    inject_vec       调用方若有连接器，可直接给一个 (1, hidden) 向量在第 site 层注入
核心是 **分块 prefill**：一次性 prefill 会把整段 logits 物化(32k 时 >20 GiB)而 OOM；
分块后 32k 只需约 8.8 GB，且快约 90 倍(见 PAPER §4.5)。
'''
from __future__ import annotations
import argparse, os, time
from typing import Dict, Optional
import torch


def chunked_prefill(model, ids, chunk: int = 2048):
    """分块 prefill，返回 (past_key_values, 秒, 峰值GB)"""
    dev = ids.device
    T = int(ids.shape[1])
    torch.cuda.reset_peak_memory_stats() if ids.is_cuda else None
    past = None; t0 = time.time()
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        am = torch.ones(1, e, dtype=torch.long, device=dev)
        pos = torch.arange(s, e, dtype=torch.long, device=dev).unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=ids[:, s:e], attention_mask=am, position_ids=pos,
                        cache_position=torch.arange(s, e, device=dev),
                        past_key_values=past, use_cache=True)
        past = out.past_key_values
    sec = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3) if ids.is_cuda else 0.0
    return past, sec, peak


def answer_long(rt, context: str, question: str, chunk: int = 2048, max_new: int = 64,
                use_memory: bool = True, inject_vec: Optional[torch.Tensor] = None,
                site: float = 0.5) -> Dict[str, object]:
    """长上下文问答。rt 是 TowerRuntime（提供 tok / model / query）。"""
    tok, model, dev = rt.tok, rt.model, rt.device
    mem_hit = None
    if use_memory:
        try:
            mem_hit = rt.query(question)
        except Exception:
            mem_hit = None
    prefix = ''
    if mem_hit:
        prefix = '【长期记忆】' + question.strip() + ' 答案：' + str(mem_hit) + chr(10)
    text = prefix + context + chr(10) + question if context else prefix + question
    ids = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                  add_generation_prompt=True, tokenize=True,
                                  return_tensors='pt', return_dict=True)['input_ids'].to(dev)
    past, sec, peak = chunked_prefill(model, ids, chunk)
    T = int(ids.shape[1])
    hooks = []
    if inject_vec is not None:
        nlayer = len(model.model.language_model.layers)
        li = int(nlayer * site)
        y = inject_vec.to(torch.bfloat16)
        def _h(_m, _i, o):
            hh = o[0] if isinstance(o, tuple) else o
            return ((hh + y,) + tuple(o[1:])) if isinstance(o, tuple) else hh + y
        hooks.append(model.model.language_model.layers[li].register_forward_hook(_h))
    try:
        am2 = torch.ones(1, T + 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            o = model.generate(input_ids=ids[:, -1:], attention_mask=am2, past_key_values=past,
                               max_new_tokens=max_new, do_sample=False, temperature=1.0,
                               top_p=1.0, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                               return_dict_in_generate=True)
        ans = tok.decode(o.sequences[0], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()
    return {'answer': ans, 'prompt_tokens': T, 'prefill_sec': sec, 'peak_gb': peak,
            'memory_hit': mem_hit}


def selftest(lens=(2400, 18000), chunk: int = 2048, needle: str = 'ZK7Q-4419',
             question: str = '这次部署用的密钥是什么？只回答密钥本身。') -> bool:
    from tower_runtime import TowerRuntime
    rt = TowerRuntime(verbose=True, m=256)
    rt.build()
    ok_all = True
    for L in lens:
        body = ['【项目档案】部署密钥：' + needle + '。']
        i = 0
        while True:
            body.append('第' + str(i) + '条日志：用户' + str(1000 + i % 9000) +
                        '访问了模块鉴权，耗时' + str(5 + i % 900) + '毫秒，返回码200。')
            i += 1
            txt = chr(10).join(body)
            if len(rt.tok(txt, add_special_tokens=False)['input_ids']) >= L or i > 20000:
                break
        # 关掉记忆前置，单纯测长上下文本身
        r = answer_long(rt, txt, question, chunk=chunk, use_memory=False, max_new=24)
        hit = needle.replace('-', '') in r['answer'].replace('-', '')
        ok_all = ok_all and hit
        print('    L=' + str(L).ljust(6) + ' tokens=' + str(r['prompt_tokens']).ljust(6) +
              ' 命中=' + str(hit) + ' prefill=' + str(round(r['prefill_sec'], 1)) + 's' +
              ' peak=' + str(round(r['peak_gb'], 2)) + 'GB  ans=' + repr(str(r['answer'])[:40]),
              flush=True)
    print('  自检结果: ' + ('通过' if ok_all else '失败'))
    return ok_all


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--lens', type=str, default='2400,18000')
    ap.add_argument('--chunk', type=int, default=2048)
    a = ap.parse_args()
    if a.selftest:
        selftest(lens=[int(x) for x in a.lens.split(',')], chunk=a.chunk)
    else:
        print('用法: python longctx.py --selftest [--lens 2400,18000]')
