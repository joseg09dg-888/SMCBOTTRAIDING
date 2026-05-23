//+------------------------------------------------------------------+
//|  SMCBotEA.mq5 — Reads Python bot signals and executes orders     |
//|  Place in: MQL5\Experts\  then Compile + Attach to EURUSD chart  |
//+------------------------------------------------------------------+
#property copyright "SMC Trading Bot"
#property version   "1.00"
#property strict

// Signal file paths (absolute)
string SIGNAL_FILE = "C:\\Users\\jose-\\projects\\trading_agent\\mt5_signals\\signal.json";
string RESULT_FILE = "C:\\Users\\jose-\\projects\\trading_agent\\mt5_signals\\result.json";
string LOCK_FILE   = "C:\\Users\\jose-\\projects\\trading_agent\\mt5_signals\\processing.lock";

//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(3);  // check every 3 seconds
   Print("SMC Bot EA started — watching for signals");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("SMC Bot EA stopped");
}

//+------------------------------------------------------------------+
void OnTimer()
{
   // Avoid double-processing
   if(FileIsExist(LOCK_FILE, FILE_COMMON)) return;
   if(!FileIsExist(SIGNAL_FILE, FILE_COMMON)) return;

   // Create lock
   int lh = FileOpen(LOCK_FILE, FILE_WRITE|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(lh == INVALID_HANDLE) return;
   FileWriteString(lh, "1");
   FileClose(lh);

   // Read signal
   int fh = FileOpen(SIGNAL_FILE, FILE_READ|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(fh == INVALID_HANDLE) { FileDelete(LOCK_FILE, FILE_COMMON); return; }

   string json = "";
   while(!FileIsEnding(fh))
      json += FileReadString(fh);
   FileClose(fh);
   FileDelete(SIGNAL_FILE, FILE_COMMON);

   Print("SMC Bot EA: signal received: ", json);

   // Parse JSON fields
   string symbol  = ParseString(json, "symbol",    "EURUSD");
   string dir     = ParseString(json, "direction", "long");
   double volume  = ParseDouble(json, "volume",    0.01);
   double sl_pts  = ParseDouble(json, "sl_pips",   50.0);
   double tp_pts  = ParseDouble(json, "tp_pips",   100.0);
   string comment = ParseString(json, "comment",   "SMC Bot");

   ENUM_ORDER_TYPE otype = (dir == "long" || dir == "LONG") ?
                            ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   // Get symbol info
   if(!SymbolSelect(symbol, true)) {
      WriteResult("{\"retcode\":4301,\"error\":\"symbol not found\",\"order\":0}");
      FileDelete(LOCK_FILE, FILE_COMMON);
      return;
   }

   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   double pt  = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int    dg  = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

   double price, sl, tp;
   if(otype == ORDER_TYPE_BUY) {
      price = ask;
      sl    = NormalizeDouble(price - sl_pts * pt * 10, dg);
      tp    = NormalizeDouble(price + tp_pts * pt * 10, dg);
   } else {
      price = bid;
      sl    = NormalizeDouble(price + sl_pts * pt * 10, dg);
      tp    = NormalizeDouble(price - tp_pts * pt * 10, dg);
   }

   // Execute trade
   MqlTradeRequest req  = {};
   MqlTradeResult  res  = {};
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = symbol;
   req.volume    = volume;
   req.type      = otype;
   req.price     = price;
   req.sl        = sl;
   req.tp        = tp;
   req.deviation = 20;
   req.magic     = 234000;
   req.comment   = comment;
   req.type_time = ORDER_TIME_GTC;
   req.type_filling = ORDER_FILLING_IOC;

   bool sent = OrderSend(req, res);

   // Write result
   string result_json = StringFormat(
      "{\"retcode\":%d,\"order\":%d,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,"
      "\"symbol\":\"%s\",\"direction\":\"%s\",\"sent\":%s}",
      res.retcode, res.order, res.price, sl, tp,
      symbol, dir, sent ? "true" : "false"
   );
   WriteResult(result_json);
   FileDelete(LOCK_FILE, FILE_COMMON);

   if(res.retcode == TRADE_RETCODE_DONE)
      Print("SMC Bot EA: trade executed! Order=", res.order, " Price=", res.price);
   else
      Print("SMC Bot EA: trade failed retcode=", res.retcode);
}

//+------------------------------------------------------------------+
void WriteResult(string json)
{
   int rh = FileOpen(RESULT_FILE, FILE_WRITE|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(rh == INVALID_HANDLE) return;
   FileWriteString(rh, json);
   FileClose(rh);
}

//+------------------------------------------------------------------+
string ParseString(string json, string key, string def)
{
   string search = "\"" + key + "\":\"";
   int start = StringFind(json, search);
   if(start < 0) return def;
   start += StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end < 0) return def;
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
double ParseDouble(string json, string key, double def)
{
   string search = "\"" + key + "\":";
   int start = StringFind(json, search);
   if(start < 0) return def;
   start += StringLen(search);
   int end = start;
   while(end < StringLen(json)) {
      ushort c = StringGetCharacter(json, end);
      if(c == ',' || c == '}') break;
      end++;
   }
   string val = StringSubstr(json, start, end - start);
   return StringToDouble(val);
}

void OnTick() {}
//+------------------------------------------------------------------+
