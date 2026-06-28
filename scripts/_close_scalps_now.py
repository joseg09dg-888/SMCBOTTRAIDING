import sys
sys.path.insert(0, '.')
import MetaTrader5 as mt5

mt5.initialize()

# Cerrar scalps - dejar swing 68803299
tickets_scalps = [68777937, 68793723, 68806317]
for t in tickets_scalps:
    pos = mt5.positions_get(ticket=t)
    if pos:
        p = pos[0]
        order_type = mt5.ORDER_TYPE_BUY if p.type == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(p.symbol).ask if p.type == 1 else mt5.symbol_info_tick(p.symbol).bid
        req = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': p.symbol,
            'volume': p.volume,
            'type': order_type,
            'position': t,
            'price': price,
            'deviation': 20,
            'magic': 234000,
            'comment': 'close_scalp',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        r = mt5.order_send(req)
        status = "CERRADO" if r and r.retcode == 10009 else f"ERROR {r.retcode if r else 'None'}"
        print(f"#{t} {p.symbol} {p.volume}L: {status}")
    else:
        print(f"#{t}: ya no existe")

mt5.shutdown()
print("Swing #68803299 GBPUSD 0.46L mantenido")
