# 个人消费信贷管理系统原型 (CreditApp) 技术设计方案

## 1. 模块目标
构建一个覆盖“开户 -> 申请 -> 审批 -> 还款 -> 监控”全生命周期的轻量级信贷后端原型。

---

## 2. 数据库建模 (SQLite)

系统采用 SQLite 存储。金额单位统一使用 **分 (Cents)**，类型为 `INTEGER`。



### 2.1 客户表 (`customers`)
存储借款人的基础身份信息。
| 字段名 | 类型 | 约束 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | 内部自增 ID |
| `name` | TEXT | NOT NULL | 客户姓名 |
| `email` | TEXT | NOT NULL UNIQUE | 唯一业务键，用于检索 |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 注册时间 |

### 2.2 贷款合同表 (`loans`)
核心业务表，管理合同条款与生命周期。
| 字段名 | 类型 | 约束 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | 贷款 ID |
| `customer_id` | INTEGER | NOT NULL, FOREIGN KEY | 关联 customers.id |
| `principal_cents`| INTEGER | NOT NULL, CHECK (> 0) | 贷款本金（分） |
| `interest_rate` | REAL | NOT NULL, CHECK (0.0-1.0) | 年化利率（如 0.05） |
| `term_months` | INTEGER | NOT NULL, CHECK (> 0) | 贷款期限（月） |
| `status` | TEXT | NOT NULL, CHECK (IN...) | 状态：`pending`, `approved`, `rejected`, `closed` |
| `applied_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 申请时间 |
| `approved_at` | TIMESTAMP | NULLABLE | 审批通过/拒绝的时间 |

### 2.3 还款记录表 (`repayments`)
记录每一笔入账流水。
| 字段名 | 类型 | 约束 | 说明 |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | 流水号 |
| `loan_id` | INTEGER | NOT NULL, FOREIGN KEY | 关联 loans.id |
| `amount_cents` | INTEGER | NOT NULL, CHECK (> 0) | 还款金额（分） |
| `paid_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 还款时间 |

---

## 3. 核心异常定义
定义三个业务异常类，用于在 `CreditApp` 内部逻辑校验失败时抛出：

* `DuplicateKeyError`: 邮箱冲突（开户失败）。
* `RecordNotFound`: ID 或 Email 不存在。
* `InvalidOperationError`: 业务逻辑违规（如：重复审批、对拒绝的贷款录入还款）。

---

## 4. 核心类：CreditApp

### 4.1 构造函数 (`__init__`)
1.  建立 `sqlite3.Connection` 持久连接。
2.  启用外键约束：`PRAGMA foreign_keys = ON;`。
3.  自动执行建表 SQL (使用 `IF NOT EXISTS`)。

### 4.2 核心业务方法 (API)

| 方法名 | 逻辑过程 | 关键约束 |
| :--- | :--- | :--- |
| `add_customer` | 插入客户记录 | 捕获唯一索引冲突并转译为 `DuplicateKeyError` |
| `apply_loan` | 关联 Email 创建贷款 | 初始状态必须为 `pending` |
| `approve_loan` | 更新审批状态及时间 | **仅允许**处理 `pending` 状态的记录 |
| `record_repayment` | 插入流水并更新主表 | 1. 仅限 `approved` 贷款；2. 计算累计还款，若足额则置为 `closed` |
| `customer_balance` | 聚合查询本金与已还金额 | 需处理该用户无贷款时的边界情况 (返回 0, 0) |
| `overdue_loans` | 筛选逾期记录 | 逻辑：`status='approved'` 且 (当前时间 - 最后还款时间) > N天 |
| `portfolio_summary` | 全盘宏观统计 | 包括：总用户数、活跃贷款、平均利率、平均本金、不良率 |

---

## 5. 开发约束与安全规范

1.  **参数化查询**：禁止使用 `f-string` 拼接 SQL，所有变量必须通过 `?` 占位符传递。
    * *示例*：`cursor.execute("SELECT... WHERE email=?", (email,))`
2.  **事务原子性**：所有写操作（尤其是 `record_repayment` 中的双表更新）必须封装在 `with self.conn:` 上下文管理器中，确保要么全部成功，要么自动回滚。
3.  **计算单位**：所有金额字段在 Python 与 DB 交互时保持 `int` (分)，仅在 `portfolio_summary` 输出报表时转换为 `float` (元)。

---
